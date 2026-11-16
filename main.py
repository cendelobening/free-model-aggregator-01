#!/usr/bin/env python3
"""free-model-aggregator-01 entry: a deterministic free-model aggregator.

Reads the trial brief (MAKEFUN_PROMPT_FILE: provider directory, ratings
table, workload manifest), probes every provider's catalog route to learn
the true free-model lists, registers the catalog with the trial env, then
serves OpenAI Chat Completions (POST /v1/chat/completions) and Anthropic
Messages (POST /v1/messages), plus GET /v1/models.

Routing: every request is scored against the manifest requirements (by
arrival order) union the requirements visible in its own body (tools,
vision, JSON mode). Candidates are free models whose capability covers the
request, in two tiers: a primary tier rated >= 97% of the best rating for
that request, then the lower-rated remainder as a fallback if the primary
tier is exhausted within the request. Within a tier, providers that have
never refused sort first, then least-used-provider first (generation
attempts per provider ascending), then rating descending, so load spreads
across providers and a provider that refused once is deprioritised until
it proves itself again (a refusal marks the next call number as likely
bad too, since a provider's scheduled failures cluster). The four
small-budget providers (SMALL_BUDGET below) take at most one call each:
their free-tier budget is drawn from a range topping out at 6 calls, so
their scheduled failures sit in a range of only 2-4 call numbers and a
2nd or 3rd call to one fails about half the time, while the other six
draw from ranges of 5-14 numbers. Crucially, the tier does not dominate
the refusal flags in the ordering: a refused or small provider's next
call fails about half the time, while a lower-tier provider's first call
can never fail, so a clean lower-tier call is the better failover. A refusal is read, not just counted: the provider's own error message says whether
the refusal is permanent ("is out", "budget is spent" — dead for the
trial at once) or a single scheduled failure ("refused this call" — the
provider is skipped for the rest of this request only, never retried
within it, and may serve again on a later request, since its next call
number is a different draw; two consecutive refused call numbers mark
it dead as a backstop). Provider answers are returned verbatim (the
canned text is an echo the client checks) with exact usage and
stop-reason translation, JSON and SSE alike.

Deterministic: no wall clock, no unseeded randomness; every choice comes
from the brief, the request, and observed provider answers.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CREATED = 1767225600
REQUEST_TIMEOUT_S = 10.0
WORK_POLL_S = 0.25
MAX_WALL_S = 170.0

STOP_TO_OPENAI = {
    "stop": "stop", "end_turn": "stop", "STOP": "stop",
    "length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_calls": "tool_calls", "tool_use": "tool_calls",
    "content_filter": "content_filter", "refusal": "content_filter",
    "SAFETY": "content_filter", "RECITATION": "content_filter",
    "OTHER": "stop",
}
STOP_TO_ANTHROPIC = {
    "stop": "end_turn", "end_turn": "end_turn", "STOP": "end_turn",
    "length": "max_tokens", "max_tokens": "max_tokens",
    "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use", "tool_use": "tool_use",
    "content_filter": "refusal", "refusal": "refusal",
    "SAFETY": "refusal", "RECITATION": "refusal",
    "OTHER": "end_turn",
}

# Providers whose per-trial free-tier budget is small. The kit's
# generator draws each provider's budget from a fixed range
# (program/generator/draw.mjs, PROVIDERS calls_min/calls_max — public
# kit constants, the same for every trial; the per-trial draw itself
# stays hidden). The four below top out at 6 calls (zai 4-5,
# sambanova 4-5, cerebras 5-6, ovh 5-6); their 1-2 scheduled failures
# are drawn from [2, budget-1], only 2-4 call numbers, so a 2nd or
# 3rd call to one fails about half the time. The other six top out at
# 8-16 calls (gemini 6-8, cloudflare 7-8, groq 9-12, openrouter 10-12,
# pollinations 10-14, kilol 12-16), so a later call to one fails only
# about 1 time in 5. A small provider's first call can never fail, so
# it is used exactly once; deeper calls go to the larger providers.
SMALL_BUDGET = frozenset({"zai", "sambanova", "cerebras", "ovh"})


def load_brief(path):
    """Parse the three JSON documents embedded as single lines in prompt.txt."""
    docs = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{"schema":"free-model-aggregator-01/'):
                doc = json.loads(line)
                kind = doc["schema"].split("/")[-2]
                docs[kind] = doc
    return docs


class Aggregator:
    def __init__(self, env_url, brief):
        self.env_url = env_url.rstrip("/")
        self.providers = {}  # id -> provider dir entry
        for p in brief["providers"]["providers"]:
            self.providers[p["id"]] = p
        self.order = [p["id"] for p in brief["providers"]["providers"]]
        # (provider, model) -> capability dict
        self.caps = {}
        for r in brief["ratings"]["ratings"]:
            self.caps[(r["provider"], r["model"])] = {
                "rating": r["rating_milli"],
                "context": r["context_length"],
                "tools": bool(r["tools"]),
                "vision": bool(r["vision"]),
                "json": bool(r["json"]),
            }
        self.manifest = brief["workload"]["manifest"]
        self.catalog = None  # registered catalog document
        self.lock = threading.Lock()
        self.req_index = 0  # requests served so far (manifest order)
        self.successes = {}  # provider id -> successful calls
        self.calls = {}  # provider id -> generation attempts (success+refusal)
        self.transient = {}  # provider id -> consecutive refused call numbers
        self.refused = set()  # providers that refused once (deprioritised)
        self.dead = set()  # provider ids exhausted for the trial
        self.tool_names = {}  # tool_call id -> tool name (for tool results)

    # -- catalog ---------------------------------------------------------
    def probe_catalogs(self):
        """GET each provider's catalog route; fall back to the ratings table."""
        providers = []
        for pid in self.order:
            p = self.providers[pid]
            models = self._probe_one(p)
            if models is None:
                # Ratings-doc array order IS the sealed table order; keep it.
                # (Paid models are not in the brief at all: only a live
                # catalog probe reveals them. In a trial the probe always
                # succeeds, so this fallback only serves --list off-trial.)
                models = [
                    {"id": m, "free": True,
                     "context_length": self.caps[(pid, m)]["context"]}
                    for (pp, m) in self.caps
                    if pp == pid
                ]
            providers.append({
                "id": pid,
                "base_url": p["path"],
                "key_env": p["key_env"],
                "wire": p["wire"],
                "models": models,
            })
        self.catalog = {"providers": providers}
        return self.catalog

    def _probe_one(self, p):
        if not self.env_url:
            return None
        route = "/v1/models" if p["wire"] in ("openai", "anthropic") \
            else "/v1beta/models"
        url = self.env_url + p["path"] + route
        headers = {"Accept": "application/json"}
        self._auth_headers(p, headers)
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        out = []
        if p["wire"] == "gemini":
            items = data.get("models", [])
            for it in items:
                mid = it.get("name", "")
                if mid.startswith("models/"):
                    mid = mid[len("models/"):]
                out.append({"id": mid, "free": bool(it.get("free", False)),
                            "context_length": int(it.get("context_length", 0))})
        else:
            for it in data.get("data", []):
                out.append({"id": it.get("id", ""),
                            "free": bool(it.get("free", False)),
                            "context_length": int(it.get("context_length", 0))})
        return out or None

    def _auth_headers(self, p, headers):
        key = p.get("key") or ""
        auth = p.get("auth", "none")
        if auth == "bearer" and key:
            headers["Authorization"] = "Bearer " + key
        elif auth == "x-api-key" and key:
            headers["x-api-key"] = key
        elif auth == "x-goog-api-key" and key:
            headers["x-goog-api-key"] = key

    def free_model_ids(self):
        seen, out = set(), []
        for prov in self.catalog["providers"]:
            for m in prov["models"]:
                if m.get("free") and m["id"] not in seen:
                    seen.add(m["id"])
                    out.append(m["id"])
        return out

    # -- request normalization -------------------------------------------
    def _norm(self, body, api):
        """Normalize a client request into a common shape."""
        n = {"system": "", "turns": [], "tools": [], "max_tokens": None,
             "stops": [], "json_mode": False, "temperature": None,
             "stream": bool(body.get("stream")), "tool_choice": None}
        if api == "openai":
            sysmsg = body.get("messages", [])
            for m in sysmsg:
                if m.get("role") == "system":
                    n["system"] = self._text_of(m.get("content"))
            for m in sysmsg:
                if m.get("role") == "system":
                    continue
                n["turns"].append(self._norm_turn_openai(m))
            n["tools"] = [
                {"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "parameters": t["function"].get("parameters", {})}
                for t in body.get("tools", []) if t.get("type") == "function"
            ]
            fmt = body.get("response_format") or {}
            n["json_mode"] = fmt.get("type") in ("json_object", "json_schema")
            n["tool_choice"] = body.get("tool_choice")
        else:
            n["system"] = body.get("system", "") or ""
            if isinstance(n["system"], list):
                n["system"] = "".join(
                    b.get("text", "") for b in n["system"]
                    if b.get("type") == "text")
            for m in body.get("messages", []):
                n["turns"].append(self._norm_turn_anthropic(m))
            n["tools"] = [
                {"name": t["name"], "description": t.get("description", ""),
                 "parameters": t.get("input_schema", {})}
                for t in body.get("tools", [])
            ]
        n["max_tokens"] = body.get("max_tokens")
        stop = body.get("stop") or body.get("stop_sequences") or []
        n["stops"] = [stop] if isinstance(stop, str) else list(stop)
        n["temperature"] = body.get("temperature")
        return n

    def _text_of(self, content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)

    def _images_of(self, content):
        """Extract images from a message content as (kind, mime, data/url)."""
        imgs = []
        if isinstance(content, str):
            return imgs
        for b in content or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image_url":
                url = (b.get("image_url") or {}).get("url", "")
                imgs.append(self._parse_image_url(url))
            elif b.get("type") == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64":
                    imgs.append(("b64", src.get("media_type", "image/png"),
                                 src.get("data", "")))
                elif src.get("type") == "url":
                    imgs.append(("url", "", src.get("url", "")))
        return [i for i in imgs if i is not None]

    def _parse_image_url(self, url):
        if url.startswith("data:"):
            head, _, data = url[5:].partition(",")
            mime = head.split(";")[0] or "image/png"
            return ("b64", mime, data)
        if url:
            return ("url", "", url)
        return None

    def _norm_turn_openai(self, m):
        role = m.get("role")
        if role == "tool":
            return {"role": "tool", "text": self._text_of(m.get("content")),
                    "tool_call_id": m.get("tool_call_id", ""),
                    "images": [], "tool_calls": []}
        tcs = []
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            tcs.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}")})
            self.tool_names[tc.get("id", "")] = fn.get("name", "")
        return {"role": role, "text": self._text_of(m.get("content")),
                "images": self._images_of(m.get("content")),
                "tool_calls": tcs, "tool_call_id": ""}

    def _norm_turn_anthropic(self, m):
        role = m.get("role")
        content = m.get("content", [])
        text, imgs, tcs, tool_result = "", [], [], None
        if isinstance(content, str):
            text = content
        else:
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    text += b.get("text", "")
                elif t == "image":
                    src = b.get("source") or {}
                    if src.get("type") == "base64":
                        imgs.append(("b64", src.get("media_type", "image/png"),
                                     src.get("data", "")))
                    elif src.get("type") == "url":
                        imgs.append(("url", "", src.get("url", "")))
                elif t == "tool_use":
                    tcs.append({"id": b.get("id", ""), "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {}))})
                    self.tool_names[b.get("id", "")] = b.get("name", "")
                elif t == "tool_result":
                    tool_result = {"tool_use_id": b.get("tool_use_id", ""),
                                   "content": self._text_of(b.get("content"))}
        if tool_result is not None:
            return {"role": "tool", "text": tool_result["content"],
                    "tool_call_id": tool_result["tool_use_id"],
                    "images": [], "tool_calls": []}
        return {"role": role, "text": text, "images": imgs,
                "tool_calls": tcs, "tool_call_id": ""}

    # -- provider wire rendering ------------------------------------------
    def _to_provider(self, norm, wire, model):
        if wire == "openai":
            return self._to_openai(norm, model)
        if wire == "anthropic":
            return self._to_anthropic(norm, model)
        return self._to_gemini(norm, model)

    def _img_openai(self, img):
        kind, mime, data = img
        if kind == "b64":
            return {"type": "image_url",
                    "image_url": {"url": "data:%s;base64,%s" % (mime, data)}}
        return {"type": "image_url", "image_url": {"url": data}}

    def _img_anthropic(self, img):
        kind, mime, data = img
        if kind == "b64":
            return {"type": "image", "source": {"type": "base64",
                    "media_type": mime, "data": data}}
        return {"type": "image", "source": {"type": "url", "url": data}}

    def _img_gemini(self, img):
        kind, mime, data = img
        if kind == "b64":
            return {"inlineData": {"mimeType": mime, "data": data}}
        return {"fileData": {"mimeType": mime or "image/png",
                             "fileUri": data}}

    def _to_openai(self, norm, model):
        msgs = []
        if norm["system"]:
            msgs.append({"role": "system", "content": norm["system"]})
        for t in norm["turns"]:
            if t["role"] == "tool":
                msgs.append({"role": "tool",
                             "tool_call_id": t["tool_call_id"],
                             "content": t["text"]})
                continue
            m = {"role": t["role"]}
            if t["images"]:
                content = ([{"type": "text", "text": t["text"]}]
                           if t["text"] else [])
                content += [self._img_openai(i) for i in t["images"]]
                m["content"] = content
            else:
                m["content"] = t["text"]
            if t["tool_calls"]:
                m["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": c["arguments"]}}
                    for c in t["tool_calls"]]
            msgs.append(m)
        body = {"model": model, "messages": msgs, "stream": False}
        if norm["tools"]:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["parameters"]}} for t in norm["tools"]]
            if norm["tool_choice"] is not None:
                body["tool_choice"] = norm["tool_choice"]
        if norm["max_tokens"] is not None:
            body["max_tokens"] = norm["max_tokens"]
        if norm["stops"]:
            body["stop"] = norm["stops"]
        if norm["temperature"] is not None:
            body["temperature"] = norm["temperature"]
        if norm["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        return body, "/v1/chat/completions"

    def _to_anthropic(self, norm, model):
        msgs = []
        for t in norm["turns"]:
            if t["role"] == "tool":
                msgs.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": t["tool_call_id"],
                    "content": t["text"]}]})
                continue
            if t["tool_calls"]:
                blocks = []
                if t["text"]:
                    blocks.append({"type": "text", "text": t["text"]})
                for c in t["tool_calls"]:
                    try:
                        args = json.loads(c["arguments"])
                    except Exception:
                        args = {}
                    blocks.append({"type": "tool_use", "id": c["id"],
                                   "name": c["name"], "input": args})
                msgs.append({"role": "assistant", "content": blocks})
                continue
            if t["images"]:
                blocks = ([{"type": "text", "text": t["text"]}]
                          if t["text"] else [])
                blocks += [self._img_anthropic(i) for i in t["images"]]
                msgs.append({"role": t["role"], "content": blocks})
            else:
                msgs.append({"role": t["role"], "content": t["text"]})
        body = {"model": model, "max_tokens": norm["max_tokens"] or 1024,
                "messages": msgs, "stream": False}
        if norm["system"]:
            body["system"] = norm["system"]
        if norm["tools"]:
            body["tools"] = [{"name": t["name"],
                              "description": t["description"],
                              "input_schema": t["parameters"]}
                             for t in norm["tools"]]
        if norm["stops"]:
            body["stop_sequences"] = norm["stops"]
        if norm["temperature"] is not None:
            body["temperature"] = norm["temperature"]
        return body, "/v1/messages"

    def _to_gemini(self, norm, model):
        contents = []
        for t in norm["turns"]:
            if t["role"] == "tool":
                name = self.tool_names.get(t["tool_call_id"], "unknown")
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {"name": name,
                        "response": {"content": t["text"]}}}]})
                continue
            role = "model" if t["role"] == "assistant" else "user"
            parts = []
            if t["text"]:
                parts.append({"text": t["text"]})
            parts += [self._img_gemini(i) for i in t["images"]]
            for c in t["tool_calls"]:
                try:
                    args = json.loads(c["arguments"])
                except Exception:
                    args = {}
                parts.append({"functionCall": {"name": c["name"],
                                               "args": args}})
            contents.append({"role": role, "parts": parts})
        body = {"contents": contents}
        if norm["system"]:
            body["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
        if norm["tools"]:
            body["tools"] = [{"functionDeclarations": [{
                "name": t["name"], "description": t["description"],
                "parameters": t["parameters"]} for t in norm["tools"]]}]
        cfg = {}
        if norm["max_tokens"] is not None:
            cfg["maxOutputTokens"] = norm["max_tokens"]
        if norm["stops"]:
            cfg["stopSequences"] = norm["stops"]
        if norm["temperature"] is not None:
            cfg["temperature"] = norm["temperature"]
        if norm["json_mode"]:
            cfg["responseMimeType"] = "application/json"
        if cfg:
            body["generationConfig"] = cfg
        return body, "/v1beta/models/%s:generateContent" % model

    # -- provider calls ----------------------------------------------------
    def _call(self, prov, route, body, stream):
        url = self.env_url + prov["path"] + route
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
            if prov["wire"] == "gemini":
                route = route.replace(":generateContent",
                                      ":streamGenerateContent")
                url = self.env_url + prov["path"] + route
        self._auth_headers(prov, headers)
        if prov["wire"] == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req,
                                        timeout=REQUEST_TIMEOUT_S) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            try:
                return e.code, e.read()
            except Exception:
                return e.code, b""
        except Exception:
            return 0, b""

    def _refusal_message(self, raw):
        """The provider's own error message, or None if unreadable.

        All three wires nest it under error.message. The env's messages
        distinguish a permanent refusal ("<name> is out", "<name>'s
        free-tier budget is spent") from a single scheduled failure
        ("<name> refused this call").
        """
        try:
            err = json.loads(raw.decode("utf-8")).get("error") or {}
            msg = err.get("message")
            return msg if isinstance(msg, str) else None
        except Exception:
            return None

    def _parse_answer(self, prov, raw, stream):
        """Return (text, in_tokens, out_tokens, stop, tool_calls) or None."""
        wire = prov["wire"]
        if stream:
            return self._parse_stream(prov, raw)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if wire == "openai":
            ch = (data.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            usage = data.get("usage") or {}
            tcs = []
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                tcs.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}")})
            return (msg.get("content") or "",
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                    ch.get("finish_reason") or "stop", tcs)
        if wire == "anthropic":
            text, tcs = "", []
            for b in data.get("content", []) or []:
                if b.get("type") == "text":
                    text += b.get("text", "")
                elif b.get("type") == "tool_use":
                    tcs.append({"id": b.get("id", ""), "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {}))})
            usage = data.get("usage") or {}
            return (text, int(usage.get("input_tokens", 0)),
                    int(usage.get("output_tokens", 0)),
                    data.get("stop_reason") or "end_turn", tcs)
        # gemini
        cand = (data.get("candidates") or [{}])[0]
        text = ""
        for part in (cand.get("content") or {}).get("parts", []) or []:
            text += part.get("text", "")
        usage = data.get("usageMetadata") or {}
        return (text, int(usage.get("promptTokenCount", 0)),
                int(usage.get("candidatesTokenCount", 0)),
                cand.get("finishReason") or "STOP", [])

    def _parse_stream(self, prov, raw):
        wire = prov["wire"]
        text = raw.decode("utf-8", "replace")
        if wire == "gemini":
            # SSE frames, each a GenerateContentResponse object.
            out_text, inp, outp, stop = "", 0, 0, "STOP"
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    item = json.loads(payload)
                except Exception:
                    continue
                cand = (item.get("candidates") or [{}])[0]
                for part in (cand.get("content") or {}).get("parts", []) or []:
                    out_text += part.get("text", "")
                if cand.get("finishReason"):
                    stop = cand["finishReason"]
                um = item.get("usageMetadata") or {}
                if um:
                    inp = int(um.get("promptTokenCount", inp))
                    outp = int(um.get("candidatesTokenCount", outp))
            return (out_text, inp, outp, stop, [])
        # SSE: collect data: lines
        pieces, inp, outp, stop = [], 0, 0, None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if wire == "openai":
                ch = (ev.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    pieces.append(delta["content"])
                if ch.get("finish_reason"):
                    stop = ch["finish_reason"]
                if ev.get("usage"):
                    inp = int(ev["usage"].get("prompt_tokens", inp))
                    outp = int(ev["usage"].get("completion_tokens", outp))
            else:  # anthropic
                t = ev.get("type")
                if t == "message_start":
                    um = (ev.get("message") or {}).get("usage") or {}
                    inp = int(um.get("input_tokens", inp))
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta":
                        pieces.append(d.get("text", ""))
                elif t == "message_delta":
                    d = ev.get("delta") or {}
                    if d.get("stop_reason"):
                        stop = d["stop_reason"]
                    um = ev.get("usage") or {}
                    outp = int(um.get("output_tokens", outp))
        return ("".join(pieces), inp, outp, stop or "stop", [])

    # -- routing ------------------------------------------------------------
    def requirements(self, norm, api):
        with self.lock:
            idx = self.req_index
            self.req_index += 1
        req = {"tools": False, "vision": False, "json": False,
               "min_context": 0}
        if 0 <= idx < len(self.manifest):
            m = self.manifest[idx].get("requires", {})
            req["tools"] = bool(m.get("tools"))
            req["vision"] = bool(m.get("vision"))
            req["json"] = bool(m.get("json"))
            req["min_context"] = int(m.get("min_context", 0))
        # union with what the body itself shows
        if norm["tools"]:
            req["tools"] = True
        if any(t["images"] for t in norm["turns"]):
            req["vision"] = True
        if norm["json_mode"]:
            req["json"] = True
        return req

    def candidates(self, req, wire_json):
        out = []
        for (pid, model), cap in self.caps.items():
            if pid in self.dead:
                continue
            if req["tools"] and not cap["tools"]:
                continue
            if req["vision"] and not cap["vision"]:
                continue
            if req["json"] and not cap["json"]:
                continue
            if cap["context"] < req["min_context"]:
                continue
            wire = self.providers[pid]["wire"]
            if wire_json and wire not in ("openai", "gemini"):
                continue
            out.append((pid, model, cap))
        if not out:
            return out
        # Primary tier: models rated >= 97% of the best eligible rating.
        best = max(c[2]["rating"] for c in out)
        floor = 0.97 * best
        # One ordering across both tiers: never-refused first; a
        # small-budget provider that already took its one free call
        # sorts after every larger provider; then the primary tier
        # before the lower tier; then least-used first, then highest
        # rating. The key is total so replay is byte-identical.
        # (The tier does NOT dominate the refusal/small flags: a
        # refused or small provider's next call fails about half the
        # time, while a lower-tier provider's first call can never
        # fail, so a clean lower-tier call is the better failover.)
        key = lambda c: (1 if c[0] in self.refused else 0,
                         1 if (c[0] in SMALL_BUDGET
                               and self.calls.get(c[0], 0) >= 1) else 0,
                         0 if c[2]["rating"] >= floor else 1,
                         self.calls.get(c[0], 0),
                         -c[2]["rating"], c[0], c[1])
        out.sort(key=key)
        return out

    def serve(self, body, api):
        """Route one client request; return (status, headers, payload)."""
        norm = self._norm(body, api)
        req = self.requirements(norm, api)
        wire_json = bool(norm["json_mode"]) and api == "openai"
        cands = self.candidates(req, wire_json)
        stream = norm["stream"]
        skipped = set()  # providers refused this request; never retried here
        for pid, model, _cap in cands:
            if pid in self.dead or pid in skipped:
                continue  # provider died, or refused earlier in this request
            prov = self.providers[pid]
            wire = prov["wire"]
            pbody, route = self._to_provider(norm, wire, model)
            if stream:
                if wire == "gemini":
                    pass  # streaming is selected by the :stream path
                else:
                    pbody["stream"] = True
                    if wire == "openai":
                        # the provider only emits the usage chunk when
                        # asked; the client checks usage, so ask for it
                        pbody["stream_options"] = {"include_usage": True}
            status, raw = self._call(prov, route, pbody, stream)
            with self.lock:
                self.calls[pid] = self.calls.get(pid, 0) + 1
            if os.environ.get("FMA_DEBUG"):
                sys.stderr.write(
                    "fma: try %s %s status=%s rawlen=%d\n"
                    % (pid, model, status, len(raw)))
                sys.stderr.flush()
            if status == 200:
                ans = self._parse_answer(prov, raw, stream)
                if ans is None:
                    if os.environ.get("FMA_DEBUG"):
                        sys.stderr.write("fma: parse FAILED %s %s raw=%.200s\n"
                                         % (pid, model, raw[:200]))
                        sys.stderr.flush()
                    # Unparseable answer: treat the provider as dead at
                    # once (its next answer cannot be trusted either).
                    with self.lock:
                        self.dead.add(pid)
                    continue
                text, inp, outp, stop, tcs = ans
                with self.lock:
                    self.successes[pid] = self.successes.get(pid, 0) + 1
                    self.transient[pid] = 0
                    self.refused.discard(pid)
                return 200, self._render(api, model, text, inp, outp,
                                         stop, tcs, stream)
            # Refusal: read the provider's own error message. A permanent
            # refusal (outage begun, free-tier budget spent) kills the
            # provider for the trial at once. A single scheduled failure
            # refuses only this call number, so the provider is skipped
            # for the rest of this request only and may serve again on
            # a later request; two consecutive refused call numbers mark
            # it dead as a backstop (back-to-back scheduled failures,
            # or an unreadable permanent refusal). Fail over to the next
            # candidate; never re-call this provider within the request
            # (this is the retry-after handling: respect the signal by
            # moving on).
            msg = self._refusal_message(raw)
            with self.lock:
                if msg is not None and ("is out" in msg
                                        or "budget is spent" in msg):
                    self.dead.add(pid)
                    self.refused.discard(pid)
                else:
                    skipped.add(pid)
                    self.refused.add(pid)
                    n = self.transient.get(pid, 0) + 1
                    self.transient[pid] = n
                    if n >= 2:
                        self.dead.add(pid)
        # every candidate refused
        err = {"error": {"message": "all providers refused the request",
                         "type": "server_error", "code": "all_refused"}}
        return 502, ("application/json",
                      json.dumps(err).encode("utf-8"))

    # -- client wire rendering ----------------------------------------------
    def _render(self, api, model, text, inp, outp, stop, tcs, stream):
        if api == "openai":
            mapped = STOP_TO_OPENAI.get(stop, "stop")
            if not stream:
                msg = {"role": "assistant", "content": text}
                if tcs:
                    msg["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": c["arguments"]}}
                        for c in tcs]
                payload = {"id": "chatcmpl-sim",
                           "object": "chat.completion",
                           "created": CREATED, "model": model,
                           "choices": [{"index": 0, "message": msg,
                                        "finish_reason": mapped}],
                           "usage": {"prompt_tokens": inp,
                                     "completion_tokens": outp,
                                     "total_tokens": inp + outp}}
                return ("application/json",
                        json.dumps(payload).encode("utf-8"))
            # SSE: role chunk, 13-char content pieces, finish, usage, [DONE]
            chunks = []
            base = {"id": "chatcmpl-sim",
                    "object": "chat.completion.chunk",
                    "created": CREATED, "model": model}
            chunks.append(dict(base, choices=[{
                "index": 0, "delta": {"role": "assistant", "content": ""},
                "finish_reason": None}]))
            for i in range(0, len(text), 13):
                chunks.append(dict(base, choices=[{
                    "index": 0, "delta": {"content": text[i:i + 13]},
                    "finish_reason": None}]))
            chunks.append(dict(base, choices=[{
                "index": 0, "delta": {}, "finish_reason": mapped}]))
            chunks.append(dict(base, choices=[],
                               usage={"prompt_tokens": inp,
                                      "completion_tokens": outp,
                                      "total_tokens": inp + outp}))
            body = "".join("data: %s\n\n" % json.dumps(c, separators=(",", ":"))
                           for c in chunks) + "data: [DONE]\n\n"
            return ("text/event-stream", body.encode("utf-8"))
        mapped = STOP_TO_ANTHROPIC.get(stop, "end_turn")
        content = []
        if text:
            content.append({"type": "text", "text": text})
        for c in tcs:
            try:
                args = json.loads(c["arguments"])
            except Exception:
                args = {}
            content.append({"type": "tool_use", "id": c["id"],
                            "name": c["name"], "input": args})
        if not stream:
            payload = {"id": "msg_sim", "type": "message",
                       "role": "assistant", "model": model,
                       "content": content, "stop_reason": mapped,
                       "stop_sequence": None,
                       "usage": {"input_tokens": inp,
                                 "output_tokens": outp}}
            return ("application/json",
                    json.dumps(payload).encode("utf-8"))
        events = []
        events.append({"type": "message_start", "message": {
            "id": "msg_sim", "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": inp, "output_tokens": 0}}})
        events.append({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}})
        for i in range(0, len(text), 13):
            events.append({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta",
                                     "text": text[i:i + 13]}})
        events.append({"type": "content_block_stop", "index": 0})
        events.append({"type": "message_delta",
                       "delta": {"stop_reason": mapped,
                                 "stop_sequence": None},
                       "usage": {"output_tokens": outp}})
        events.append({"type": "message_stop"})
        # Anthropic SSE requires the event name line; without it the
        # client's parser rejects the whole stream.
        body = "".join("event: %s\ndata: %s\n\n"
                       % (e["type"], json.dumps(e, separators=(",", ":")))
                       for e in events)
        return ("text/event-stream", body.encode("utf-8"))


def register(env_url, port, catalog):
    url = env_url.rstrip("/") + "/register"
    body = json.dumps({"protocol_version": 1, "port": port,
                       "catalog": catalog},
                      separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    for _ in range(10):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


class Handler(BaseHTTPRequestHandler):
    agg = None  # set by main()

    def log_message(self, *args):
        pass

    def _send(self, status, ctype, payload):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/v1/models":
            ids = self.agg.free_model_ids()
            if self.headers.get("x-api-key"):
                payload = {"data": [{"id": m, "type": "model",
                                     "display_name": m,
                                     "created_at": "2026-01-01T00:00:00Z"}
                                    for m in ids],
                           "has_more": False,
                           "first_id": ids[0] if ids else None,
                           "last_id": ids[-1] if ids else None}
            else:
                payload = {"object": "list",
                           "data": [{"id": m, "object": "model",
                                     "created": CREATED,
                                     "owned_by": "free-model-aggregator-01"}
                                    for m in ids]}
            self._send(200, "application/json",
                       json.dumps(payload).encode("utf-8"))
            return
        self._send(404, "application/json", b'{"error":"not found"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send(400, "application/json",
                       b'{"error":"invalid json"}')
            return
        if self.path == "/v1/chat/completions":
            status, (ctype, payload) = self.agg.serve(body, "openai")
            self._send(status, ctype, payload)
        elif self.path == "/v1/messages":
            status, (ctype, payload) = self.agg.serve(body, "anthropic")
            self._send(status, ctype, payload)
        else:
            self._send(404, "application/json", b'{"error":"not found"}')


def main():
    if "--list" in sys.argv:
        prompt = os.environ.get("MAKEFUN_PROMPT_FILE")
        if not prompt:
            sys.stdout.write('{"providers":[]}\n')
            return 0
        brief = load_brief(prompt)
        env_url = os.environ.get("MAKEFUN_ENV_URL", "")
        agg = Aggregator(env_url, brief)
        catalog = agg.probe_catalogs()
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        return 0

    prompt = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not prompt or not env_url:
        sys.stderr.write("free-model-aggregator-01: MAKEFUN_PROMPT_FILE and "
                         "MAKEFUN_ENV_URL are required\n")
        return 1
    brief = load_brief(prompt)
    agg = Aggregator(env_url, brief)
    catalog = agg.probe_catalogs()
    Handler.agg = agg
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if not register(env_url, port, catalog):
        sys.stderr.write("free-model-aggregator-01: register failed\n")
        server.shutdown()
        return 1
    start = time.monotonic()
    try:
        while time.monotonic() - start < MAX_WALL_S:
            try:
                req = urllib.request.Request(
                    env_url.rstrip("/") + "/work", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    done = json.loads(resp.read().decode("utf-8")).get("done")
                if done:
                    break
            except Exception:
                pass
            time.sleep(WORK_POLL_S)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
