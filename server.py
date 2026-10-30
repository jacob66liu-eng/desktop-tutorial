from __future__ import annotations

import json
import os
import random
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.json"
LOCK = threading.RLock()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


class OllamaError(Exception):
    pass


def ollama_chat(messages, temperature=0.4):
    request_data = json.dumps({"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                               "options": {"temperature": temperature}}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL + "/api/chat", data=request_data,
                                    headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["message"]["content"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        raise OllamaError(f"无法连接 Ollama（{OLLAMA_URL}，模型 {OLLAMA_MODEL}）。请确认 Ollama 正在运行且已安装该模型。") from exc


def now():
    return datetime.now(timezone.utc)


def stamp(value=None):
    return (value or now()).isoformat()


def load_data():
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            data.setdefault("questions", [])
            max_id = max((int(q.get("id", 0)) for q in data["questions"]), default=0)
            data["next_id"] = max(int(data.get("next_id", 1)), max_id + 1)
            data.setdefault("message_count", 0)
            data.setdefault("permission_pending", False)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"questions": [], "next_id": 1, "message_count": 0, "permission_pending": False}


def save_data(data):
    tmp = DATA_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_FILE)


def rewrite_question(raw):
    result = ollama_chat([
        {"role": "system", "content": "你负责把用户写下的复盘问题改写成面向用户的第二人称问题。将原文中指向用户自己的‘我’改为‘你’，相关人称保持一致；调整提问句式并进行自然的同义改写；严格保留核心意思，不补充信息、不回答问题。只输出一个改写后的问题，不加引号、解释或前缀。"},
        {"role": "user", "content": raw},
    ], temperature=0.2)
    return result.strip().strip('"“”')


def question_by_id(data, qid):
    return next((q for q in data["questions"] if q["id"] == qid), None)


def format_question(q, with_answer=False):
    lines = [f"📝复盘题：{q['rewritten']}" ]
    if with_answer:
        lines.append(f"✅参考答案：{q['answer'] if q.get('answer') else '尚未录入答案'}")
        if q.get("tags"):
            lines.append("🏷标签：" + "、".join(t["name"] for t in q["tags"]))
    return "\n".join(lines)


def draw(data, mode="all"):
    questions = data["questions"]
    if not questions:
        return "当前问题库暂无题目"
    if mode == "tagged":
        pool = [q for q in questions if q.get("tags")]
    elif mode == "untagged":
        pool = [q for q in questions if not q.get("tags")]
    else:
        pool = questions
    return format_question(random.choice(pool or questions))


def exact_command(text):
    value = text.strip()
    if value in ("??", "??A"):
        return ("random", value == "??A")
    match = re.fullmatch(r"\?\?(A)?(\d+)", value)
    if match:
        return ("id", (int(match.group(2)), bool(match.group(1))))
    return None


def answer_match(text):
    match = re.fullmatch(r"\s*A(\d+)\s*[:：]\s*(.+?)\s*", text, re.S)
    return (int(match.group(1)), match.group(2)) if match else None


def explicit_answer_save_request(text):
    value = text.strip()
    return bool(re.search(r"保存.{0,12}(?:答案|回答)|(?:答案|回答).{0,12}保存|记录.{0,12}(?:答案|回答)", value))


def add_question(data, original):
    rewritten = rewrite_question(original)
    qid = data["next_id"]
    data["next_id"] += 1
    data["questions"].append({"id": qid, "original": original, "rewritten": rewritten,
                              "answer": None, "answer_history": [], "pending_answer": None,
                              "tags": [], "created_at": stamp(), "last_untagged_reminder": None})
    return qid


def process_message(data, text):
    command = exact_command(text)
    if command:
        kind, value = command
        if kind == "random":
            if value:
                q = random.choice(data["questions"]) if data["questions"] else None
                return format_question(q, True) if q else "当前问题库暂无题目"
            return draw(data)
        qid, with_answer = value
        q = question_by_id(data, qid)
        return format_question(q, with_answer) if q else "找不到这个问题ID"

    # A final answer replacement requires a pending answer and explicit confirmation.
    confirm = re.fullmatch(r"\s*！(.+?)！\s*", text, re.S)
    if confirm:
        q = next((item for item in data["questions"] if item.get("pending_answer")), None)
        if q:
            q["pending_answer"] = confirm.group(1)
            return f"将把问题ID：{q['id']} 的答案更新为：{q['pending_answer']}\n请回复“确认覆盖”执行，或“取消覆盖”保留旧答案。"
        return "当前没有待确认的答案修改。"
    if text.strip() == "确认覆盖":
        q = next((item for item in data["questions"] if item.get("pending_answer")), None)
        if q:
            q.setdefault("answer_history", []).append({"answer": q.get("answer"), "saved_at": stamp()})
            q["answer"] = q.pop("pending_answer")
            return "答案已更新，之前的版本已保留。"
        return "当前没有待确认的答案修改。"
    if text.strip() == "取消覆盖":
        for q in data["questions"]:
            if q.get("pending_answer"):
                q["pending_answer"] = None
                return "已取消覆盖，原答案保留。"
    answer = answer_match(text)
    if answer:
        q = question_by_id(data, answer[0])
        if not q:
            return "找不到这个问题ID"
        if q.get("answer"):
            return f"问题ID：{q['id']} 上一个版本的答案是：{q['answer']}\n请先和我讨论最终版本，再用“！最终答案内容！”提交；我会请你确认后覆盖。"
        q["answer"] = answer[1]
        tags = re.findall(r'"([^"\n]+)"', answer[1])
        for name in tags:
            q["tags"].append({"name": name, "source": "user", "created_at": stamp()})
        q["answer"] = re.sub(r'\s*"[^"\n]+"\s*', " ", answer[1]).strip()
        return "答案已保存。" + (" 已按引号内容添加标签。" if tags else "")

    if explicit_answer_save_request(text):
        return "这段回答要保存到哪个问题ID？"

    mark = re.search(r"\?\?\s*$", text)
    if mark and text.strip() != "??":
        original = text[:mark.start()]
        try:
            qid = add_question(data, original)
            return f"已录入问题库，问题ID：{qid}"
        except OllamaError as exc:
            return str(exc)

    return None


def reminders(data):
    due = []
    current = now()
    for q in data["questions"]:
        if q.get("tags"):
            continue
        created = datetime.fromisoformat(q["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        previous = q.get("last_untagged_reminder")
        if current - created < timedelta(hours=72):
            continue
        if previous:
            if random.random() >= 0.30:
                continue
        due.append(q)
    if due:
        q = random.choice(due)
        q["last_untagged_reminder"] = stamp()
        return f"⚠提醒：问题ID：{q['id']} 已经入库超过3天，还没有分配标签，请问需要给它分配标签吗？"
    return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        encoded = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if urlparse(self.path).path == "/api/state":
            with LOCK:
                data = load_data()
                self._send(200, json.dumps(data, ensure_ascii=False))
            return
        path = ROOT / ("index.html" if self.path in ("/", "/index.html") else self.path.lstrip("/"))
        if path.is_file() and path.parent == ROOT:
            self._send(200, path.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path == "/api/tag":
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                qid, action = int(payload["id"]), payload["action"]
            except (ValueError, KeyError, json.JSONDecodeError):
                self._send(400, json.dumps({"error": "invalid request"})); return
            with LOCK:
                data = load_data()
                q = question_by_id(data, qid)
                if not q:
                    self._send(404, json.dumps({"error": "找不到这个问题ID"}, ensure_ascii=False)); return
                if action == "add" and str(payload.get("name", "")).strip():
                    name = str(payload["name"]).strip()
                    if not any(t["name"] == name for t in q["tags"]):
                        q["tags"].append({"name": name, "source": "user", "created_at": stamp()})
                elif action in ("rename", "delete"):
                    old = str(payload.get("old", ""))
                    tag = next((t for t in q["tags"] if t["name"] == old), None)
                    if tag:
                        # User-initiated edits are explicit; agent edits require separate consent.
                        if action == "rename" and str(payload.get("name", "")).strip():
                            tag["name"] = str(payload["name"]).strip()
                        elif action == "delete":
                            q["tags"].remove(tag)
                else:
                    self._send(400, json.dumps({"error": "invalid action"})); return
                save_data(data)
                self._send(200, json.dumps(data, ensure_ascii=False))
            return
        if self.path != "/api/message":
            self._send(404, json.dumps({"error": "not found"})); return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            text = str(payload.get("text", ""))
            history = payload.get("history", [])
        except (ValueError, json.JSONDecodeError):
            self._send(400, json.dumps({"error": "invalid request"})); return
        with LOCK:
            data = load_data()
            response = process_message(data, text)
            countable = exact_command(text) is None and not text.strip().startswith(("确认覆盖", "取消覆盖")) and not re.fullmatch(r"\s*！.+！\s*", text, re.S)
            accepted = data.get("permission_pending", False)
            normalized = text.strip().lower()
            positive = normalized in ("是", "好", "可以", "愿意", "yes", "y", "同意") or normalized.startswith(("当然", "愿意", "好呀", "好的", "没问题", "可以啊", "来一道"))
            negative = normalized in ("不", "不用", "不方便", "拒绝", "no", "n") or normalized.startswith(("现在不", "不太方便", "改天", "不用了"))
            if accepted and positive:
                mode = random.choices(["tagged", "all", "untagged"], weights=[60, 20, 20])[0]
                response = draw(data, mode)
                data["permission_pending"] = False
                countable = False
            elif accepted and negative:
                data["permission_pending"] = False
                response = "好的，我们继续聊。"
                countable = False
            if response is None:
                safe_history = []
                if isinstance(history, list):
                    for item in history[-40:]:
                        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
                            safe_history.append({"role": item["role"], "content": item["content"][:8000]})
                try:
                    response = ollama_chat([
                        {"role": "system", "content": "你是 Jacob 专属的问题复盘助手。用自然、简短、体贴的中文回应普通聊天。不要替 Jacob 编造或代写他自己的复盘答案；答案只由他本人通过 A数字：内容 提供。不要自行把聊天内容存入问题库或答案。标签由用户在界面中管理；如果你提出新增标签，必须明确告知这是独立操作并先征求同意。复盘指令由系统程序处理，不要声称已执行未执行的存储操作。"},
                        *safe_history,
                        {"role": "user", "content": text[:8000]},
                    ])
                except OllamaError as exc:
                    response = str(exc)
            if countable:
                data["message_count"] += 1
            permission = False
            if data["message_count"] >= 15:
                data["message_count"] = 0
                data["permission_pending"] = True
                permission = True
            reminder = None
            ordinary_chat = (not exact_command(text) and not answer_match(text)
                             and not text.rstrip().endswith("??")
                             and not text.strip().startswith(("确认覆盖", "取消覆盖"))
                             and not re.fullmatch(r"\s*！.+！\s*", text, re.S)
                             and not (accepted and (positive or negative)))
            if ordinary_chat:
                reminder = reminders(data)
            save_data(data)
            self._send(200, json.dumps({"response": response, "permission": permission, "reminder": reminder,
                                      "state": data}, ensure_ascii=False))

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Jacob 复盘助手已启动：http://127.0.0.1:8765")
    server.serve_forever()
