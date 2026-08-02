import pandas as pd
import json
from rapidfuzz import fuzz
from dateutil import parser
import os

model_outputs = {
    "Nemotron": "outputs",
    "Cohere":"outputs2",
    "Qwen": "outputs3"
}

ground=pd.read_csv('Groundtruth.csv')

def normalize_date(date):
    try:
        return parser.parse(str(date), dayfirst=True).strftime("%Y-%m-%d")
    except:
        return ""
def normalize_amount(amount):
    try:
        return float(str(amount).replace(",", ""))
    except:
        return None
def normalize_currency(curr):
    curr = str(curr).strip().upper()

    mapping = {
        "RS": "Rs",
        "RS.": "Rs",
        "₹": "Rs",
        "INR": "Rs",
        "Rs.": "Rs",
        "":"Rs"
    }

    return mapping.get(curr, curr)
def normalize_invoice(inv):
    return str(inv).strip().lstrip("0").upper()
for model_name,outputs_folder in model_outputs.items():
    vendor_correct=0
    date_correct=0
    amount_correct=0
    curr_correct=0
    inv_correct=0
    gst_correct=0
    for _, row in ground.iterrows():

        receipt_id = row["Receipt ID"]

        json_path = os.path.join(outputs_folder, f"{receipt_id}.json")

        with open(json_path, "r") as f:
            prediction = json.load(f)

        vendor_similarity=fuzz.partial_ratio(prediction["vendor"].strip().upper(), row["Vendor"].strip().upper())
        
        if vendor_similarity>=90:
            vendor_correct+=1
        gt_date = normalize_date(row["Date"])
        pred_date = normalize_date(prediction["date"])

        if gt_date == pred_date:
            date_correct += 1

        gt_amount = normalize_amount(row["Amount"])
        pred_amount = normalize_amount(prediction["amount"])

        if gt_amount == pred_amount:
            amount_correct += 1

        gt_curr = normalize_currency(row["Currency"])
        pred_curr = normalize_currency(prediction["currency"])

        if gt_curr == pred_curr:
            curr_correct += 1
        gt_inv = normalize_invoice(row["Invoice no"])
        pred_inv = normalize_invoice(prediction["invoice_no"])

        if gt_inv == pred_inv:
            inv_correct += 1
        gt_gst = "" if pd.isna(row["Gst"]) else str(row["Gst"]).strip()
        pred_gst = "" if pd.isna(prediction.get("gst")) else str(prediction.get("gst")).strip()

        if gt_gst == pred_gst:
            gst_correct += 1

        

    vendor_acc=(vendor_correct/len(ground))*100
    print(f"{model_name} Vendor Accuracy: {vendor_acc:.2f}%")

    date_acc=(date_correct/len(ground))*100
    print(f"{model_name} Date Accuracy: {date_acc:.2f}%")

    amtt_acc=(amount_correct/len(ground))*100
    print(f"{model_name} Amount Accuracy: {amtt_acc:.2f}%")

    curr_acc=(curr_correct/len(ground))*100
    print(f"{model_name} Currency Accuracy: {curr_acc:.2f}%")

    inv_acc=(inv_correct/len(ground))*100
    print(f"{model_name} Invoice Accuracy: {inv_acc:.2f}%")

    gst_acc=(gst_correct/len(ground))*100
    print(f"{model_name} GST Accuracy: {gst_acc:.2f}%")
    print("*****************")


    # cost analysis 

    MODEL_USAGE = {
    "Nemotron Nano 12B VL": {
        "tokens": 129000,
        "price_per_million": 0.24   #from OpenRouter pricing
    },

    "Command-A-Vision-07-2025": {
        "tokens": 121000,
        "price_per_million": 12.50   #from Cohere pricing
    },

    "Qwen3.6-27B": {
        "tokens": 16100,
        "price_per_million": 0.90   #from Groq pricing
    }
}


def calculate_cost(tokens, price_per_million):
    return (tokens / 1_000_000) * price_per_million


for model, data in MODEL_USAGE.items():
    cost = calculate_cost(
        data["tokens"],
        data["price_per_million"]
    )

    print(model)
    print("Total tokens:", data["tokens"])
    print("Cost: $", round(cost, 6))
    print(cost*(100/12))
