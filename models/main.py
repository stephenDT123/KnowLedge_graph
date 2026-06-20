from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(title="我的第一个FastAPI")
# 数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None
# 路由
@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}