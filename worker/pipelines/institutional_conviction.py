import requests
import xml.etree.ElementTree as ET
from collections import defaultdict

HEADERS = {"User-Agent": "Vaultline Hackathon Project aidanjm369@gmail.com"}

TICKER_CUSIPS = {
    "GLD":  "78463V107",
    "SLV":  "46428Q109",
    "PPLT": "003260106",
    "PALL": "003262102",
    "DBB":  "46140H700",
    "COPX": "37954Y830",
    "TLT":  "464287432",
    "IEF":  "464287440",
    "LQD":  "464287242",
    "HYG":  "464288513",
    "MBB":  "464288588",
    "VNQ":  "922908553",
    "IFRA": "46435U713",
    "BIZD": "92189F411"
}

CUSIP_TO_CATEGORY = {
    "78463V107": "precious_metals",
    "46428Q109": "precious_metals",
    "003260106": "precious_metals",
    "003262102": "precious_metals",
    "46140H700": "industrial_metals",
    "37954Y830": "industrial_metals",
    "464287432": "sovereign_debt",
    "464287440": "sovereign_debt",
    "464287242": "corporate_credit",
    "464288513": "corporate_credit",
    "464288588": "structured_credit",
    "922908553": "real_assets",
    "46435U713": "real_assets",
    "92189F411": "alt_financing"
}

ALL_CATEGORIES = ["precious_metals", "industrial_metals", "sovereign_debt",
                   "corporate_credit", "structured_credit", "real_assets", "alt_financing"]

INVESTORS = {
    "berkshire": "0001067983",
    "bridgewater": "0001350694",
    "blackrock": "0001364742",
    "vanguard": "0000102909",
    "renaissance": "0001037389"
}


def get_matched_holdings_for_investor(cik, headers, ticker_cusips):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    forms = data["filings"]["recent"]["form"]
    accession_numbers = data["filings"]["recent"]["accessionNumber"]

    for i, form in enumerate(forms):
        if form == "13F-HR":
            break

    accession_no_dashes = accession_numbers[i].replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/index.json"
    index_response = requests.get(index_url, headers=headers)
    index_data = index_response.json()

    holdings_filename = None
    for item in index_data["directory"]["item"]:
        if item["name"].endswith(".xml") and item["name"] != "primary_doc.xml":
            holdings_filename = item["name"]
            break

    holdings_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{holdings_filename}"
    holdings_response = requests.get(holdings_url, headers=headers)

    root = ET.fromstring(holdings_response.content)
    ns = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}

    matched = []
    for info_table in root.findall("ns:infoTable", ns):
        cusip = info_table.find("ns:cusip", ns).text
        if cusip in ticker_cusips.values():
            value = int(info_table.find("ns:value", ns).text)
            shares = int(info_table.find("ns:shrsOrPrnAmt/ns:sshPrnamt", ns).text)
            matched.append({"cusip": cusip, "value": value, "shares": shares})

    return matched


def get_institutional_conviction_scores():
    """
    Fetches latest 13F holdings for all tracked investors, matches against
    our proxy tickers, and returns a 0-100 conviction score per category.
    """
    all_results = {}
    for name, cik in INVESTORS.items():
        all_results[name] = get_matched_holdings_for_investor(cik, HEADERS, TICKER_CUSIPS)

    category_totals = defaultdict(int)
    for matches in all_results.values():
        for m in matches:
            category = CUSIP_TO_CATEGORY.get(m["cusip"])
            if category:
                category_totals[category] += m["value"]

    max_total = max(category_totals.values()) if category_totals else 0

    scores = {}
    for category in ALL_CATEGORIES:
        total = category_totals.get(category, 0)
        scores[category] = round((total / max_total) * 100, 1) if max_total > 0 else 0

    return scores


# Quick manual test — only runs if you execute this file directly,
# won't run when the worker imports get_institutional_conviction_scores()
if __name__ == "__main__":
    result = get_institutional_conviction_scores()
    print("=== Institutional Conviction Scores ===")
    for category, score in result.items():
        print(f"{category}: {score}")