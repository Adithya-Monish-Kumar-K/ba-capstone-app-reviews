"""Single source of truth for the apps in the study, shared by every pipeline script."""

APPS = {
    "swiggy":     {"name": "Swiggy",     "package": "in.swiggy.android",                      "domain": "Food & Grocery"},
    "zomato":     {"name": "Zomato",     "package": "com.application.zomato",                 "domain": "Food & Grocery"},
    "blinkit":    {"name": "Blinkit",    "package": "com.grofers.customerapp",                "domain": "Food & Grocery"},
    "dominos":    {"name": "Domino's",   "package": "com.Dominos",                            "domain": "Food & Grocery"},
    "myntra":     {"name": "Myntra",     "package": "com.myntra.android",                     "domain": "Shopping"},
    "flipkart":   {"name": "Flipkart",   "package": "com.flipkart.android",                   "domain": "Shopping"},
    "amazon":     {"name": "Amazon",     "package": "in.amazon.mShop.android.shopping",       "domain": "Shopping"},
    "meesho":     {"name": "Meesho",     "package": "com.meesho.supply",                      "domain": "Shopping"},
    "paytm":      {"name": "Paytm",      "package": "net.one97.paytm",                        "domain": "Payments"},
    "phonepe":    {"name": "PhonePe",    "package": "com.phonepe.app",                        "domain": "Payments"},
    "googlepay":  {"name": "Google Pay", "package": "com.google.android.apps.nbu.paisa.user", "domain": "Payments"},
}

APP_NAMES = [a["name"] for a in APPS.values()]
DOMAINS = ["Food & Grocery", "Shopping", "Payments"]
APP_DOMAIN = {a["name"]: a["domain"] for a in APPS.values()}

# 11 apps never share one panel: charts facet by domain, and each app takes the validated
# categorical slot (blue, orange, aqua, yellow) for its position inside its domain.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
APP_COLORS = {}
for _domain in DOMAINS:
    for _i, _name in enumerate(n for n in APP_NAMES if APP_DOMAIN[n] == _domain):
        APP_COLORS[_name] = SLOTS[_i]
DOMAIN_COLORS = dict(zip(DOMAINS, SLOTS))
