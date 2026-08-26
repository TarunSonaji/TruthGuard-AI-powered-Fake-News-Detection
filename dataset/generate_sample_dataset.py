"""
dataset/generate_sample_dataset.py
-----------------------------------
TruthGuard is designed to train on the Kaggle "Fake and Real News Dataset"
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset),
which ships two CSV files: Fake.csv and True.csv.

Since that dataset requires a Kaggle account/download, this script generates
a small SYNTHETIC stand-in dataset with the same column structure
(title, text, subject, date) so the whole pipeline (train_model.py -> app.py)
can be run and demoed immediately without any manual download.

>>> IMPORTANT <<<
This synthetic data is only good enough for demonstrating that the code
works end-to-end. For a real, production-quality classifier you MUST
replace dataset/Fake.csv and dataset/True.csv with the real Kaggle files
(same filenames, same columns) and re-run train_model.py.

Usage:
    python dataset/generate_sample_dataset.py
"""

import os
import random
import pandas as pd

random.seed(42)

REAL_TEMPLATES = [
    "The {org} announced on {day} that it will invest {amount} in {topic} "
    "infrastructure over the next {years} years, according to officials "
    "familiar with the plan.",
    "Researchers at {org} published a peer-reviewed study in {journal} "
    "showing measurable progress on {topic}, the university confirmed in "
    "a statement.",
    "The city council voted {votes} in favor of a new {topic} policy "
    "that will take effect next quarter, following months of public "
    "hearings.",
    "According to data released by {org}, unemployment in the {region} "
    "region fell to {percent}% last month, continuing a two-year trend.",
    "{org} reported quarterly earnings that matched analyst expectations, "
    "with revenue from {topic} rising {percent}% year over year.",
    "Officials at the {org} confirmed that the new {topic} regulations "
    "were finalized after a public comment period that ended last week.",
    "A government audit of {org} found no major irregularities in its "
    "{topic} spending for the past fiscal year, auditors said.",
    "The health department in {region} recommended residents follow "
    "updated {topic} guidelines issued by national health authorities.",
]

FAKE_TEMPLATES = [
    "You won't believe what {org} is secretly hiding about {topic} — "
    "insiders reveal the shocking truth that the mainstream media refuses "
    "to report!!!",
    "BREAKING: Anonymous sources claim {org} is planning to control "
    "{topic} using a secret {topic} chip that doctors don't want you to "
    "know about.",
    "Scientists HATE this one trick that cures {topic} overnight, and "
    "{org} is desperately trying to ban it before everyone finds out.",
    "Leaked documents allegedly prove {org} rigged the {topic} results, "
    "according to a viral post shared thousands of times on social media.",
    "Shocking video appears to show {org} officials admitting they faked "
    "the entire {topic} crisis to control the population.",
    "A miracle {topic} remedy discovered by a small-town grandmother is "
    "being suppressed by {org}, sources close to the family claim.",
    "Is {org} secretly working with aliens on {topic}? Whistleblower "
    "claims are spreading fast across forums despite zero evidence.",
    "This one weird {topic} secret {org} does NOT want you to see — "
    "share before it gets deleted forever!",
]

ORGS = ["the Department of Transportation", "Google", "the World Health Organization",
        "the Federal Reserve", "Stanford University", "the local school board",
        "NASA", "the state legislature", "a major pharmaceutical company",
        "the United Nations", "the Ministry of Health", "City Hall"]

TOPICS = ["climate", "healthcare", "education", "renewable energy", "vaccine",
          "artificial intelligence", "public transit", "the economy",
          "cybersecurity", "housing", "immigration", "elections"]

REGIONS = ["Midwest", "Pacific Northwest", "Southeast", "national", "metro",
           "Northeast", "Gulf Coast"]

JOURNALS = ["Nature", "The Lancet", "Science", "the Journal of Public Policy"]


def _fill(template):
    return template.format(
        org=random.choice(ORGS),
        topic=random.choice(TOPICS),
        day=random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        amount=f"${random.randint(1, 900)} million",
        years=random.randint(2, 10),
        journal=random.choice(JOURNALS),
        votes=f"{random.randint(5,9)}-{random.randint(0,4)}",
        region=random.choice(REGIONS),
        percent=round(random.uniform(0.5, 9.9), 1),
    )


def build_dataframe(n_per_class: int = 600) -> (pd.DataFrame, pd.DataFrame):
    fake_rows = []
    real_rows = []
    for i in range(n_per_class):
        real_text = _fill(random.choice(REAL_TEMPLATES)) + " " + _fill(random.choice(REAL_TEMPLATES))
        fake_text = _fill(random.choice(FAKE_TEMPLATES)) + " " + _fill(random.choice(FAKE_TEMPLATES))

        real_rows.append({
            "title": real_text.split(".")[0][:80],
            "text": real_text,
            "subject": random.choice(["politicsNews", "worldnews", "technology", "health"]),
            "date": f"{random.randint(1,12)}/{random.randint(1,28)}/2023",
        })
        fake_rows.append({
            "title": fake_text.split(".")[0][:80],
            "text": fake_text,
            "subject": random.choice(["News", "conspiracy", "left-news", "politics"]),
            "date": f"{random.randint(1,12)}/{random.randint(1,28)}/2023",
        })

    return pd.DataFrame(fake_rows), pd.DataFrame(real_rows)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fake_df, real_df = build_dataframe(n_per_class=600)

    fake_path = os.path.join(out_dir, "Fake.csv")
    true_path = os.path.join(out_dir, "True.csv")

    fake_df.to_csv(fake_path, index=False)
    real_df.to_csv(true_path, index=False)

    print(f"Generated synthetic sample dataset:")
    print(f"  {fake_path}  ({len(fake_df)} rows)")
    print(f"  {true_path}  ({len(real_df)} rows)")
    print("\nReplace these files with the real Kaggle dataset for production use.")
