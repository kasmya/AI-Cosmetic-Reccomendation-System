"""Load test for the API.   Run (server already running on :8000):

    locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000 --csv loadtest/out

Traffic mix (weights): consultations 60, product score 15, product detail + review summary 10, catalog search 5,
feedback after a consultation 10. Photo endpoints (face analysis, try-on) are added when scikit-image is installed
(it supplies a public-domain NASA portrait), weight 4 and 2: they are 10-50x heavier than the rest.
"""
import random

from locust import HttpUser, between, task

SKINS = ["Dry", "Oily", "Combination", "Normal"]
CONCERNS = [[], ["acne"], ["dryness"], ["redness", "sensitivity"], ["fine_lines", "dark_spots"], ["dullness"]]
BUDGETS = [None, None, 60, 120, 200]

try:                                                          # optional photo traffic
    import cv2
    from skimage import data as skdata
    PHOTO = cv2.imencode(".jpg", cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR))[1].tobytes()
except Exception:                                             # noqa: BLE001
    PHOTO = None


class Shopper(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        r = self.client.get("/skincare/products?limit=400", name="/skincare/products")
        self.product_ids = [p["Product_id"] for p in r.json()] or [1]
        self.last = None

    @task(60)
    def consult(self):
        body = {"skin_type": random.choice(SKINS), "concerns": random.choice(CONCERNS), "per_category": 2,
                "total_budget": random.choice(BUDGETS), "colour_preferences": random.choice([[], ["warm"], ["berry"]])}
        with self.client.post("/api/v1/consult", json=body, name="/api/v1/consult", catch_response=True) as r:
            if r.status_code == 200:
                self.last = r.json()
            else:
                r.failure(f"status {r.status_code}")

    @task(15)
    def score(self):
        pid = random.choice(self.product_ids)
        self.client.get(f"/api/v1/products/{pid}/score", params={"skin_type": random.choice(SKINS)},
                        name="/api/v1/products/{id}/score")

    @task(10)
    def detail(self):
        pid = random.choice(self.product_ids)
        self.client.get(f"/api/v1/products/{pid}", name="/api/v1/products/{id}")
        self.client.get(f"/api/v1/products/{pid}/reviews/summary", name="/api/v1/products/{id}/reviews/summary")

    @task(5)
    def search(self):
        self.client.get("/api/v1/catalog/search", params={"q": random.choice(["cream", "sun", "wash", "serum"])},
                        name="/api/v1/catalog/search")

    @task(10)
    def feedback(self):
        if not self.last or not self.last.get("request_id"):
            return
        item = self.last["skincare"]["Cleanser"][0]
        self.client.post("/api/v1/feedback", json={"product_id": item["product_id"], "helpful": random.random() < 0.7,
                                                   "request_id": self.last["request_id"]}, name="/api/v1/feedback")

    @task(4)
    def face_analyze(self):
        if PHOTO:
            self.client.post("/api/v1/face/analyze", data=PHOTO, name="/api/v1/face/analyze")

    @task(2)
    def tryon(self):
        if PHOTO:
            self.client.post("/api/v1/makeup/tryon", data=PHOTO, name="/api/v1/makeup/tryon",
                             params={"lips_hex": "#a23b5e", "blush_hex": "#ee7f6a"})
