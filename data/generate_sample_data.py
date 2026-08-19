"""
Generates data/sample_data.csv — a synthetic-but-realistic balanced dataset
(2,000 articles) so the whole project can be trained and demoed immediately
without a Kaggle account.

Real articles are built to *read* like sourced wire reports (measured tone,
attribution, numbers, official titles). Fake articles are built to *read*
like sensational / conspiratorial clickbait (all-caps emphasis, absolutist
claims, emotional triggers, no attribution).

For a production-grade model, replace this with the real ISOT dataset
(see data/README.md) — this generator is a bootstrap, not a replacement.
"""
import csv
import random
import os

random.seed(42)

TOPICS = [
    "the economy", "the new healthcare bill", "the presidential election",
    "climate policy", "the education budget", "the national vaccine rollout",
    "the tech industry", "immigration reform", "the housing market",
    "the defense budget", "a new trade agreement", "the state of the judiciary",
    "renewable energy investment", "the central bank's interest rate decision",
    "the upcoming census", "a major merger between two airlines",
    "a new labor law", "the space program's next mission",
    "a public transportation overhaul", "the agriculture subsidy program",
]

REAL_SOURCES = ["Reuters", "the Associated Press", "a Treasury Department spokesperson",
                "the Bureau of Labor Statistics", "the Congressional Budget Office",
                "the Department of Health", "a spokesperson for the White House",
                "the National Weather Service", "the Federal Reserve", "a court filing"]

REAL_TEMPLATES = [
    "Officials confirmed on {day} that {topic} will move forward after a review by {source}, "
    "according to a statement released this week. The measure passed with a vote of {n1} to {n2}, "
    "and is expected to take effect over the next {n3} months. Analysts at several independent "
    "research firms said the impact would likely be gradual, citing prior similar measures.",
    "A report published by {source} on {topic} shows a change of {pct}% compared to last year. "
    "The data was compiled from {n1} separate regional offices and reviewed by independent "
    "economists before release. Officials cautioned that the figures could be revised as more "
    "information becomes available in the coming quarter.",
    "Lawmakers debated {topic} for several hours on {day} before reaching a preliminary agreement. "
    "{source} said the next formal vote is scheduled within {n3} weeks. Several committee members "
    "requested additional data before final approval, and a public comment period will follow.",
    "According to {source}, {topic} has been under review since early this year, with a formal "
    "decision expected by the end of the {n3}-month evaluation period. The agency said it would "
    "publish its full findings once the review concludes, in line with standard procedure.",
]

FAKE_TEMPLATES = [
    "YOU WON'T BELIEVE what they don't want you to know about {topic}!!! Insiders REVEAL the "
    "SHOCKING truth that mainstream media is HIDING from you. Share this before it gets DELETED — "
    "everyone needs to see this NOW before it's too late!",
    "BREAKING: Secret documents EXPOSE the real agenda behind {topic}. A whistleblower who wishes "
    "to remain anonymous says this changes EVERYTHING. Wake up, people — they have been LYING to "
    "us for years and this PROVES it beyond any doubt!",
    "Doctors HATE this one simple fact about {topic} that could change your life forever! The "
    "government has been covering this up for decades and NOW the truth is finally coming out. "
    "Forward this to everyone you know before they take it down!",
    "SHOCKING new evidence about {topic} that NO ONE is talking about! A source close to the "
    "situation says this is just the tip of the iceberg. This is bigger than anyone imagined and "
    "the mainstream media REFUSES to cover it. Wake up and share NOW!",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def make_real():
    t = random.choice(REAL_TEMPLATES)
    return t.format(
        day=random.choice(DAYS), topic=random.choice(TOPICS),
        source=random.choice(REAL_SOURCES),
        n1=random.randint(2, 400), n2=random.randint(1, 300),
        n3=random.randint(1, 24), pct=round(random.uniform(0.5, 12.5), 1),
    )


def make_fake():
    t = random.choice(FAKE_TEMPLATES)
    return t.format(topic=random.choice(TOPICS))


def main():
    rows = []
    for _ in range(1000):
        rows.append({"text": make_real(), "label": "REAL"})
    for _ in range(1000):
        rows.append({"text": make_fake(), "label": "FAKE"})
    random.shuffle(rows)

    out_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
