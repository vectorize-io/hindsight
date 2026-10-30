"""The rest of a real bank: ~300 memories every recall case is searched through.

A case with four memories tests almost nothing about ranking: any order that keeps
the right fact in the top three passes, so a ranker that only fuses the arms (RRF)
and a cross-encoder score the same. A real bank holds hundreds of facts and recall
reranks up to ``HINDSIGHT_API_RERANKER_MAX_CANDIDATES`` (300) of them, so every
case also gets this pool: one person's ordinary life and work, written to sit
NEXT TO each case without answering it — other trips, other error codes, order
ids one character off, other quarters' budgets, other people's jobs and pets,
other services' databases, other outages and their causes.

Two rules keep it fair, and ``tests/test_recall_haystack.py`` checks both:

* no memory names a word a case's grading looks for (it would pass a case it
  should not, or fail one by outranking the answer with a second right answer);
* no memory is dated inside a window a temporal case asks about, since anything
  done "yesterday" IS an answer to "what did I do yesterday?".

Stored as written (``chunks`` extraction): no model call, one fact per memory,
dated by ``mentioned_at``, which is the temporal arm's fallback date.
"""

from __future__ import annotations

import datetime
import os
import random

#: How many pool memories each case's bank gets; 0 turns the pool off (the old
#: four-memory banks). Default: the reranker's default candidate cap.
HAYSTACK_SIZE = int(os.getenv("HINDSIGHT_EVAL_RECALL_HAYSTACK", "300"))

#: Windows a temporal case asks about: last-month (March 2024), chinese-last-week,
#: yesterday. named-week (first week of March 2024) sits inside the first.
BLACKOUTS = (
    (datetime.date(2024, 3, 1), datetime.date(2024, 3, 31)),
    (datetime.date(2024, 5, 8), datetime.date(2024, 5, 16)),
    (datetime.date(2024, 6, 13), datetime.date(2024, 6, 17)),
)

_TRIPS = [
    f"I went to {city} {why}."
    for city, why in (
        ("Madrid", "for a product conference"),
        ("Prague", "to see an old university friend"),
        ("Vienna", "for my cousin's wedding"),
        ("Porto", "for a long weekend of hiking"),
        ("Athens", "for a client workshop"),
        ("Copenhagen", "for the design week"),
        ("Edinburgh", "for the festival"),
        ("Krakow", "to visit my grandmother"),
        ("Seville", "for a short holiday"),
        ("Dublin", "for a partner meeting"),
        ("Amsterdam", "for a hackathon"),
        ("Zurich", "for a security training"),
        ("Budapest", "for a friend's birthday"),
        ("Helsinki", "for a sauna weekend"),
        ("Marseille", "for a sailing course"),
        ("Munich", "for the team offsite"),
        ("Tallinn", "for a startup meetup"),
        ("Valencia", "to watch a football match"),
        ("Bruges", "for a day trip"),
        ("Florence", "for a cooking class"),
    )
] + [
    "My train to Milan was delayed by three hours because of a strike.",
    "My connecting flight in Frankfurt was moved to the next morning.",
    "The ferry to Mallorca was cancelled because of high winds.",
    "I lost my luggage on the way back from Stockholm.",
    "I booked a hotel in Barcelona for the autumn.",
]

_ERRORS = [
    f"{feature} fails with ERR_{code} {when}."
    for feature, code, when in (
        ("Checkout", "7713_TIMEOUT", "when the tax service does not answer"),
        ("Checkout", "7371_TIMEOUT", "when the inventory lock is held too long"),
        ("Checkout", "1773_CART", "when the cart has more than 99 items"),
        ("Payments", "7137_DECLINED", "when the card issuer rejects the charge"),
        ("Payments", "3177_TIMEOUT", "when the fraud check is slow"),
        ("Login", "7317_LOCKED", "after five wrong passwords"),
        ("Login", "1737_SSO", "when the identity provider certificate is stale"),
        ("Search", "7173_QUERY", "when the query has unbalanced quotes"),
        ("Search", "3717_SHARD", "when a shard is offline"),
        ("Upload", "7771_SIZE", "when the file is over 2 GB"),
        ("Upload", "1377_TYPE", "for files with an unknown extension"),
        ("Export", "7714_TIMEOUT", "when the report spans more than a year"),
        ("Export", "3771_FORMAT", "when a column name has a comma"),
        ("Notifications", "7313_QUOTA", "once the SMS quota is used up"),
        ("Signup", "1733_EMAIL", "when the email domain has no MX record"),
        ("Invoices", "7371_PDF", "when the logo is a huge PNG"),
        ("Sync", "7133_CONFLICT", "when two devices edit the same note"),
        ("Webhooks", "7737_GIVEUP", "after the receiver answers 500 ten times"),
    )
]

_ORDERS = [
    f"Order {order} {what}."
    for order, what in (
        ("A9-44-ZO", "shipped from the Hamburg warehouse on Tuesday"),
        ("A9-44-ZR", "is waiting for a missing part"),
        ("A9-45-ZQ", "shipped from the Antwerp warehouse on Friday"),
        ("A9-43-ZQ", "was returned by the customer as the wrong size"),
        ("A8-44-ZQ", "was delivered to the wrong address"),
        ("A9-44-QQ", "was cancelled by the customer before shipping"),
        ("A9-44-ZZ", "shipped by express from the Lyon warehouse"),
        ("B9-44-ZQ", "was refunded after a payment dispute"),
        ("A9-04-ZQ", "is held at customs in Dover"),
        ("A9-44-Z0", "shipped from the Gdansk warehouse on Monday"),
        ("Q9-44-ZA", "was split into two parcels"),
        ("A9-77-ZQ", "was returned because the box arrived empty"),
    )
]

_PEOPLE = [
    "Dr. Whitfield reviewed the safety report and asked for one more trial.",
    "Dr. Oyelowo reviewed the audit report and signed it off.",
    "Dr. Okafor-Wright chaired the ethics board meeting.",
    "Dr. Whitford presented the clinical results to the board.",
    "Dr. Smithson reviewed the travel policy.",
    "Dr. Lindqvist approved the lab equipment purchase.",
    "Dr. Achterberg asked for the raw data from the second study.",
    "Professor Kowalczyk reviewed the grant application.",
    "My friend Carla works at Hooli as a product designer.",
    "My brother Luca started a new job at Vandelay Industries.",
    "My old manager Paul now works at Soylent Corp.",
    "Carla's partner works at Umbrella Labs in research.",
    "Diego Ferreira works at Stark Robotics as a firmware engineer.",
    "Stark Robotics has its headquarters in Lyon.",
    "Umbrella Labs has its headquarters in Basel.",
    "Hooli has its headquarters in Palo Alto.",
    "Vandelay Industries has its headquarters in Newark.",
    "Alice Martin works at Wayne Analytics.",
    "Wayne Analytics opened a new office in Gothenburg.",
    "Acme Logistics, a different company from the robotics one, is based in Turin.",
    "My brother Kenji has a parrot called Pip.",
    "My friend Sofia adopted a greyhound called Pepper.",
    "My cousin Hiro has two rabbits, Kuro and Shiro.",
    "My colleague Anna's dog is a beagle called Rufus.",
    "My mother has a cat called Olive.",
    "My sister-in-law lives in Osaka and teaches violin.",
    "My aunt lives in Nagoya and keeps bees.",
    "My uncle's dog Bruno is a bernese mountain dog.",
]

_OUTAGES = [
    "The login outage on 12 April was caused by a bad config push.",
    "The bad config push happened because the review step was skipped on a Friday.",
    "The checkout slowdown on 20 May was caused by a missing database index.",
    "The payments outage on 2 June was caused by the card network being down.",
    "The email delays on 8 July were caused by our provider throttling us.",
    "The dashboard outage on 14 August was caused by a memory leak in the chart service.",
    "The memory leak in the chart service came from an unbounded cache.",
    "The upload outage on 3 September was caused by an expired storage credential.",
    "The storage credential expired because nobody owned the service account.",
    "The search outage on 11 October was caused by a runaway reindex job.",
    "The mobile app crash on 21 November was caused by a null user avatar.",
    "The DNS outage on 2 December was caused by a typo in a zone file.",
]

_SERVICES = [
    "The payments service uses Redis for its job queue.",
    "We moved the search service from Solr to Elasticsearch.",
    "The inventory service stores its data in DynamoDB.",
    "We decided to keep SQLite for the desktop app.",
    "The analytics pipeline writes to ClickHouse.",
    "We switched the notification service from RabbitMQ to Kafka.",
    "The orders dashboard reads from a nightly warehouse copy in BigQuery.",
    "We considered MongoDB for the catalog service but rejected it.",
    "The auth service caches sessions in Memcached.",
    "The reporting service moved from Airflow to Dagster.",
    "The production API keys are rotated every 90 days by the security team.",
    "The VPN certificates are rotated every year.",
    "The admin dashboard password is stored in the team vault.",
    "The CI deploy token expires every 7 days.",
    "The guest network in the office is reset every Monday.",
]

_BUDGETS = [
    "The marketing budget for Q4 2024 is $15,000.",
    "The marketing budget for Q3 2023 was $11,000.",
    "The marketing budget for Q1 2024 was $9,800.",
    "The sales budget for Q3 2024 is $9,500.",
    "The events budget for Q3 2024 is $4,200.",
    "The marketing team spent $1,350 on ads in July 2024.",
    "The hiring budget for Q3 2024 is $40,000.",
    "The design budget for Q2 2024 was $7,500.",
    "The travel budget for Q3 2024 is $6,000.",
    "The marketing budget request for Q1 2025 is $18,000.",
]

_ACTIVITIES = [
    "I went to the gym for an upper body workout.",
    "I went running along the river for 10 km.",
    "I had a haircut at the barber on Elm Street.",
    "I went to the cinema to see a documentary about whales.",
    "I had dinner with my parents at a Thai restaurant.",
    "I spent the afternoon fixing the bike chain.",
    "I went to a pottery class.",
    "I had a check-up with my GP.",
    "I went to the theatre to see Hamlet.",
    "I visited the modern art museum.",
    "I played tennis with Kenji.",
    "I went to a board game night at Sofia's place.",
    "I spent the day painting the spare bedroom.",
    "I went to a farmers' market and bought cheese.",
    "I watched the football final at a pub.",
    "I went swimming at the public pool.",
    "I had a video call with my cousin Hiro.",
    "I baked sourdough bread for the first time.",
    "I went to an opera performance of La Traviata.",
    "I went to a stand-up comedy show.",
    "I had my car serviced.",
    "I went to a climbing gym with colleagues.",
    "I went to a wine tasting.",
    "I volunteered at the food bank.",
    "I went to my nephew's school play.",
]

_WORK = [
    "We deployed version 3 of the search service.",
    "We hired Tomasz as a frontend engineer.",
    "The office moved to the fifth floor.",
    "We deployed the new onboarding flow.",
    "We hired Leila as a data scientist.",
    "We ran the quarterly planning workshop.",
    "We migrated the mobile app to the new design system.",
    "We deprecated version 1 of the public API.",
    "We shipped the CSV import feature.",
    "We hired Ravi as an engineering manager.",
    "We deployed version 2 of the notification service.",
    "We closed the Series B funding round.",
    "We held the company all-hands in the big auditorium.",
    "We rolled back the payments release after a spike in errors.",
    "We started the SOC 2 audit.",
    "We deployed dark mode to the mobile app.",
    "We ran a two-day hackathon.",
    "We signed the contract with the new cloud provider.",
    "We switched the team standup to the afternoon.",
    "We released the public status page.",
]

_EVENTS = [
    "The 2026 AI summit will be held in Paris.",
    "The 2026 climate report will be published in March.",
    "The 2025 developer summit was held in Seoul.",
    "The 2024 biodiversity summit was held in Cali.",
    "The 2026 energy summit has not chosen a city yet.",
    "The 2023 climate march in our city drew ten thousand people.",
    "The 2026 Winter Olympics are in Milan and Cortina.",
    "The 2025 robotics fair was held in Hanover.",
]

_TASTES = [
    "I love roasted cauliflower with tahini.",
    "I cooked mushroom risotto on Sunday.",
    "I tried a Georgian restaurant and loved the khachapuri.",
    "I eat oatmeal with berries for breakfast.",
    "I made a big pot of lentil soup for the week.",
    "I prefer email over chat for anything important.",
    "I prefer dark themes in every app.",
    "I prefer morning meetings to afternoon ones.",
    "I like to keep my desk completely empty.",
    "I prefer short commit messages in the imperative mood.",
    "I prefer reviewing code on paper printouts once a month.",
    "I listen to podcasts while cooking.",
    "My favourite season is autumn.",
    "I keep a paper notebook for every project.",
    "I use a mechanical keyboard with brown switches.",
]

_TOOLS = [
    "I extended my gym membership for another year.",
    "I signed up for a Spanish course on Tuesday evenings.",
    "I bought a new laptop with 32 GB of memory.",
    "I replaced the kitchen tap.",
    "I switched my phone plan to a cheaper one.",
    "I started using a password manager for the whole family.",
    "I set up automatic backups of my photos.",
    "I sold my old camera online.",
    "I bought noise-cancelling headphones for the office.",
    "I started learning the piano with an app.",
]

_LANGUAGES = [
    "Ho comprato un libro di cucina a Bologna.",
    "Ho comprato una lampada nuova per lo studio.",
    "Mio fratello lavora in una banca a Genova.",
    "Sabato abbiamo mangiato il sushi da Kenzo.",
    "Ho venduto la mia vecchia moto a un amico.",
    "La settimana scorsa ho riparato la lavatrice.",
    "Mia madre ha comprato un divano blu.",
    "Ho prenotato le vacanze in Sardegna.",
    "我在广州见了李经理，讨论了预算。",
    "我在深圳参加了一个产品发布会。",
    "我和张老师吃了午饭，聊了课程安排。",
    "我在杭州见了陈总，谈了合作。",
    "我在家里学习了一整天。",
    "我去医院做了体检。",
    "我在成都见了刘医生。",
    "我和朋友去爬山了。",
]

_MORE = [
    "We decided to run the user research sessions remotely.",
    "We agreed to freeze new features for the last two weeks of December.",
    "We chose Figma for the new brand guidelines.",
    "We decided to drop support for Internet Explorer.",
    "We switched the company laptops from Intel to Apple silicon.",
    "We decided to publish the API changelog every Friday.",
    "We picked Grafana for the on-call dashboards.",
    "We moved the design reviews to Thursdays.",
    "We decided to write all RFCs in the shared docs folder.",
    "We agreed that every incident gets a written review within a week.",
    "The mobile team ships a release every two weeks.",
    "The data team owns the nightly revenue report.",
    "The support team answers tickets within four hours on weekdays.",
    "The platform team maintains the Terraform modules.",
    "The security team runs a phishing drill every quarter.",
    "The finance team closes the books on the fifth working day.",
    "The legal team reviews every new vendor contract.",
    "The growth team is testing a new pricing page.",
    "The design team is redrawing all the app icons.",
    "The QA team automated the regression suite for the mobile app.",
    "My neighbour plays the trumpet every evening at seven.",
    "My landlord is replacing the windows in the spring.",
    "My parents celebrated their fortieth wedding anniversary.",
    "My nephew started primary school.",
    "My best friend moved to Vancouver for a job in film.",
    "My grandmother turned ninety and threw a big party.",
    "My cousin Hiro got married in a small ceremony.",
    "My brother Kenji bought a sailing boat.",
    "My friend Sofia opened a bakery near the station.",
    "My colleague Anna won the company chess tournament.",
    "I read a novel about a lighthouse keeper.",
    "I finished a course on statistics.",
    "I started journaling every evening.",
    "I gave a talk about accessibility at a local meetup.",
    "I wrote a blog post about migrating to a monorepo.",
    "I ran my first half marathon in under two hours.",
    "I planted tomatoes and basil on the balcony.",
    "I learned to make fresh pasta at home.",
    "I repaired the garden fence after the storm.",
    "I donated my old clothes to a charity shop.",
    "The car insurance went up by 12 percent this year.",
    "The electricity bill was much higher in January.",
    "The rent increases by 3 percent next year.",
    "I opened a savings account for the house deposit.",
    "I paid off the last instalment on the car loan.",
    "The quarterly sales review showed growth in Spain.",
    "The customer survey ranked onboarding as the top complaint.",
    "The churn rate dropped to 2 percent in the second quarter.",
    "The new pricing page increased signups by 8 percent.",
    "The support backlog fell under fifty tickets for the first time.",
    "The Android app reached a 4.6 rating in the store.",
    "The website's page load time dropped below one second.",
    "The quarterly security scan found two medium issues.",
    "The partner integration with the accounting app went live.",
    "The annual employee survey scored manager support highest.",
    "The board approved the plan to open an office in Madrid.",
    "Release Birch adds two-factor login for admins.",
    "Release Aspen fixed the calendar export on iOS.",
    "Release Cedar adds bulk editing of contacts.",
    "Release Maple removed the legacy reports page.",
    "Release Willow adds a public API for invoices.",
    "Release Spruce fixed the slow search on large accounts.",
    "The team migrated the email service to a new provider.",
    "The team added audit logs to the admin panel.",
    "The team rewrote the pricing calculator.",
    "The team moved the docs site to a static generator.",
    "The team shipped push notifications on Android.",
    "The team reduced the test suite runtime by forty percent.",
    "I can never remember where I parked at the airport.",
    "I keep my passport in the drawer under the printer.",
    "The spare house key is with the neighbour downstairs.",
    "The router is in the hallway cupboard.",
    "The bike lock code is written in my notebook.",
    "The boiler gets serviced every October.",
    "The dentist reminder comes by text message.",
    "The recycling is collected on Wednesdays.",
    "The library books are due back at the end of the month.",
    "The car's MOT test is due in September.",
    "The cat sitter comes twice a day when we travel.",
    "The plants need watering every three days in summer.",
    "I prefer aisle seats on long flights.",
    "I am allergic to penicillin.",
]

_TEAMMATES = [
    "Ana's favourite terminal is Kitty.",
    "Ana's favourite keyboard layout is Colemak.",
    "Ana's favourite font for code is JetBrains Mono.",
    "Ana's favourite browser is Firefox.",
    "Ana uses Neovim plugins for git blame.",
    "Ben's favourite terminal is iTerm.",
    "Ben's favourite language is Rust.",
    "Ben's favourite shell is fish.",
]

#: Every pool memory.
POOL: list[str] = (
    _TRIPS
    + _ERRORS
    + _ORDERS
    + _PEOPLE
    + _OUTAGES
    + _SERVICES
    + _BUDGETS
    + _ACTIVITIES
    + _WORK
    + _EVENTS
    + _TASTES
    + _TOOLS
    + _LANGUAGES
    + _MORE
    + _TEAMMATES
)


def _dates(count: int) -> list[str]:
    """Spread over 2023-2025, densest in 2024 where the temporal cases ask, never in a blackout."""
    rng = random.Random(20240405)
    start = datetime.date(2023, 1, 1)
    dates: list[str] = []
    while len(dates) < count:
        # Triangular around mid-2024: most of the pool sits near the cases' query dates.
        day = start + datetime.timedelta(days=int(rng.triangular(0, 1095, 540)))
        if not any(lo <= day <= hi for lo, hi in BLACKOUTS):
            dates.append(f"{day.isoformat()}T{rng.randint(8, 21):02d}:00:00")
    return dates


def haystack_items(size: int = HAYSTACK_SIZE) -> list[dict[str, object]]:
    """The first ``size`` pool memories as retain items, one document each, dated."""
    pool = POOL[:size]
    return [
        {
            "content": text,
            "document_id": f"haystack-{i}",
            "timestamp": timestamp,
        }
        for i, (text, timestamp) in enumerate(zip(pool, _dates(len(pool))))
    ]
