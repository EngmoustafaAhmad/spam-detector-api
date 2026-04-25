{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e6be2567-d410-4c15-bc9f-68c24990d705",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import pickle\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "vectorizer = pickle.load(open(\"vectorizer.pkl\", \"rb\"))\n",
    "\n",
    "models = {\n",
    "    \"NaiveBayes\": pickle.load(open(\"nb.pkl\", \"rb\")),\n",
    "    \"LogisticRegression\": pickle.load(open(\"lr.pkl\", \"rb\")),\n",
    "    \"SVM\": pickle.load(open(\"svc.pkl\", \"rb\")),\n",
    "    \"RandomForest\": pickle.load(open(\"rfc.pkl\", \"rb\")),\n",
    "    \"DecisionTree\": pickle.load(open(\"dtc.pkl\", \"rb\"))\n",
    "}\n",
    "\n",
    "# ✅ Request model\n",
    "class MessageRequest(BaseModel):\n",
    "    text: str\n",
    "\n",
    "\n",
    "@app.post(\"/predict\")\n",
    "def predict(data: MessageRequest):\n",
    "    text = data.text.lower()\n",
    "    vector = vectorizer.transform([text])\n",
    "\n",
    "    results = {}\n",
    "\n",
    "    for name, model in models.items():\n",
    "        pred = model.predict(vector)[0]\n",
    "        results[name] = \"Ham\" if pred == 1 else \"Spam\"\n",
    "\n",
    "    return results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4ab8072-283e-4f8b-9c2d-7a5e428ebea3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
