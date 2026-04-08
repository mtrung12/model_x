import json
import os
from typing import Optional


class UserStore:
    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path
        self.users: dict[str, dict] = {}

    def load(self):
        if self.store_path and os.path.exists(self.store_path):
            self.users = {}
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    self.users[item["user_id"]] = item

    def save(self):
        if not self.store_path:
            return
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for user in self.users.values():
                f.write(json.dumps(user, ensure_ascii=False) + "\n")

    def add(self, user_id: str, posts_raw: str, posts_embedding: list[float]):
        self.users[user_id] = {
            "user_id": user_id,
            "posts_raw": posts_raw,
            "posts_embedding": posts_embedding,
        }

    def get(self, user_id: str) -> Optional[dict]:
        return self.users.get(user_id)

    def get_all(self) -> list[dict]:
        return list(self.users.values())

    def __len__(self):
        return len(self.users)
