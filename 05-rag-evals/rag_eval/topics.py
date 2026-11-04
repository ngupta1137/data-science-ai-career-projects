"""Authored facts for the generated teaching corpus.

Everything here is invented for this repository. The company, the policies, and the
numbers are fictional. They exist so a retrieval-augmented question answering system can
be evaluated with known answers, offline, with no third-party data.

Each topic has one or two facts. `sentence` holds the policy text with {v1} and {v2}
placeholders. `v1_old` is the value from an archived version, used for the stale-document
question. `q_lex` reuses the document's own words. `q_para[0]` uses the colloquial alias
that appears only in the FAQ page. `q_para[1]` uses neither the formal name nor the alias.
`q_unans` asks about something the corpus never states, on a topic it does cover.
"""

from __future__ import annotations

COMPANY = "Northwind Logistics"

TOPICS: list[dict] = [
    {
        "id": "parental_leave",
        "name": "Parental Leave",
        "alias": "newborn time off",
        "sentence": "Eligible employees receive {v1} of fully paid leave after the birth or adoption of a child.",
        "v1": "16 weeks",
        "v1_old": "12 weeks",
        "q_lex": [
            "How long is the paid leave under the Parental Leave Policy?",
            "What is the length of paid leave in the Parental Leave Policy?",
        ],
        "q_para": [
            "How much newborn time off does a new parent get?",
            "How many weeks can a new mother or father stay home with full pay?",
        ],
        "q_current": "What is the current length of paid leave under the Parental Leave Policy?",
        "q_unans": "How many times can an employee use the Parental Leave Policy over a career?",
    },
    {
        "id": "remote_stipend",
        "name": "Remote Work Stipend",
        "alias": "work-from-home allowance",
        "sentence": "Remote employees receive {v1} per year toward home office equipment and must file the claim within {v2} of purchase.",
        "v1": "$1,200",
        "v2": "60 days",
        "v1_old": "$800",
        "q_lex": [
            "What is the yearly amount in the Remote Work Stipend Policy?",
            "How much does the Remote Work Stipend Policy provide each year?",
        ],
        "q_para": [
            "How big is the work-from-home allowance each year?",
            "How much money for a home desk setup does the company give annually?",
        ],
        "q2": "How soon after buying equipment must a remote employee file the stipend claim?",
        "q_current": "What is the current yearly amount in the Remote Work Stipend Policy?",
        "q_unans": "Which vendors are approved for buying remote work equipment?",
    },
    {
        "id": "travel_per_diem",
        "name": "Travel Per Diem",
        "alias": "daily travel meal money",
        "sentence": "Employees on overnight business travel receive {v1} per day for meals and incidentals.",
        "v1": "$75",
        "v1_old": "$60",
        "q_lex": [
            "What is the daily amount under the Travel Per Diem Policy?",
            "How much does the Travel Per Diem Policy pay per day?",
        ],
        "q_para": [
            "How much daily travel meal money do staff get on a trip?",
            "What does the company give for food each day when someone sleeps away on business?",
        ],
        "q_current": "What is the current daily amount under the Travel Per Diem Policy?",
        "q_unans": "Does the Travel Per Diem Policy cover alcohol purchases?",
    },
    {
        "id": "expense_deadline",
        "name": "Expense Report Deadline",
        "alias": "reimbursement filing cutoff",
        "sentence": "Expense reports must be submitted within {v1} of the purchase date, and reports above {v2} need director approval.",
        "v1": "30 days",
        "v2": "$2,500",
        "q_lex": [
            "What is the submission window in the Expense Report Deadline Policy?",
            "How long do employees have to submit expense reports under the Expense Report Deadline Policy?",
        ],
        "q_para": [
            "What is the reimbursement filing cutoff?",
            "How many days after buying something can someone still ask to be paid back?",
        ],
        "q2": "Above what amount does an expense report need director approval?",
        "q_unans": "Which app is used to file expense reports?",
    },
    {
        "id": "laptop_refresh",
        "name": "Laptop Refresh",
        "alias": "computer upgrade schedule",
        "sentence": "Company laptops are replaced every {v1}.",
        "v1": "36 months",
        "v1_old": "48 months",
        "q_lex": [
            "How often are laptops replaced under the Laptop Refresh Policy?",
            "What is the replacement cycle in the Laptop Refresh Policy?",
        ],
        "q_para": [
            "What is the computer upgrade schedule?",
            "How many months does a work machine stay with an employee before it is swapped?",
        ],
        "q_current": "What is the current replacement cycle in the Laptop Refresh Policy?",
        "q_unans": "Which laptop models does the company issue?",
    },
    {
        "id": "training_budget",
        "name": "Annual Training Budget",
        "alias": "learning money",
        "sentence": "Each employee has {v1} per year for courses and conferences.",
        "v1": "$2,000",
        "q_lex": [
            "What is the yearly amount in the Annual Training Budget Policy?",
            "How much does the Annual Training Budget Policy give each employee per year?",
        ],
        "q_para": [
            "How much learning money does each person get?",
            "What can a worker spend yearly on classes and industry events?",
        ],
        "q_unans": "Which conferences are pre-approved under the Annual Training Budget Policy?",
    },
    {
        "id": "oncall_allowance",
        "name": "On-Call Allowance",
        "alias": "pager pay",
        "sentence": "Engineers on the on-call rotation receive {v1} for each week they carry the pager.",
        "v1": "$300",
        "v1_old": "$250",
        "q_lex": [
            "What is the weekly payment in the On-Call Allowance Policy?",
            "How much does the On-Call Allowance Policy pay for each week?",
        ],
        "q_para": [
            "How much is pager pay for a week?",
            "What extra money do engineers earn for being reachable at night for seven days?",
        ],
        "q_current": "What is the current weekly payment in the On-Call Allowance Policy?",
        "q_unans": "How many engineers are on the on-call rotation each week?",
    },
    {
        "id": "badge_replacement",
        "name": "Security Badge Replacement",
        "alias": "lost access card",
        "sentence": "A lost badge must be reported within {v1}, and the replacement fee is {v2}.",
        "v1": "24 hours",
        "v2": "$25",
        "q_lex": [
            "What is the reporting window in the Security Badge Replacement Policy?",
            "How quickly must a lost badge be reported under the Security Badge Replacement Policy?",
        ],
        "q_para": [
            "How fast must a lost access card be reported?",
            "How many hours do I have to tell someone that I can no longer find my door pass?",
        ],
        "q2": "What is the fee charged for a replacement security badge?",
        "q_unans": "Where is the badge office located?",
    },
    {
        "id": "visitor_access",
        "name": "Visitor Access",
        "alias": "guest check-in",
        "sentence": "Visitors must be registered at least {v1} before arrival and may stay on site for no more than {v2} per visit.",
        "v1": "1 business day",
        "v2": "8 hours",
        "q_lex": [
            "How early must visitors be registered under the Visitor Access Policy?",
            "What is the advance registration time in the Visitor Access Policy?",
        ],
        "q_para": [
            "How far ahead must guest check-in be arranged?",
            "How much notice do I need to give before a friend can come into the building?",
        ],
        "q2": "What is the longest a visitor may stay on site during one visit?",
        "q_unans": "Do visitors need to wear a lanyard on site?",
    },
    {
        "id": "password_rotation",
        "name": "Password Rotation",
        "alias": "credential refresh",
        "sentence": "Passwords for company systems must be changed every {v1}.",
        "v1": "90 days",
        "v1_old": "180 days",
        "q_lex": [
            "How often must passwords be changed under the Password Rotation Policy?",
            "What is the rotation interval in the Password Rotation Policy?",
        ],
        "q_para": [
            "How often is the credential refresh due?",
            "After how many days must I pick a new login secret for work systems?",
        ],
        "q_current": "What is the current rotation interval in the Password Rotation Policy?",
        "q_unans": "How many characters must a company password contain?",
    },
    {
        "id": "data_retention",
        "name": "Data Retention",
        "alias": "how long we hold client files",
        "sentence": "Customer records are kept for {v1} after an account closes and then deleted.",
        "v1": "7 years",
        "v1_old": "5 years",
        "q_lex": [
            "How long are customer records kept under the Data Retention Policy?",
            "What is the retention period in the Data Retention Policy?",
        ],
        "q_para": [
            "How long we hold client files after they leave?",
            "For how many years does the firm keep what it knows about a former customer?",
        ],
        "q_current": "What is the current retention period in the Data Retention Policy?",
        "q_unans": "Who approves deleting customer records early under the Data Retention Policy?",
    },
    {
        "id": "incident_reporting",
        "name": "Incident Reporting",
        "alias": "breach notification window",
        "sentence": "Security incidents must be reported to the response team within {v1} of discovery.",
        "v1": "4 hours",
        "q_lex": [
            "How quickly must incidents be reported under the Incident Reporting Policy?",
            "What is the reporting deadline in the Incident Reporting Policy?",
        ],
        "q_para": [
            "What is the breach notification window?",
            "Within how many hours must I raise the alarm after I spot something suspicious?",
        ],
        "q_unans": "Which phone number reaches the security response team?",
    },
    {
        "id": "referral_bonus",
        "name": "Referral Bonus",
        "alias": "recruiting tip reward",
        "sentence": "Employees receive {v1} when a referred candidate completes {v2} of employment.",
        "v1": "$3,000",
        "v2": "6 months",
        "v1_old": "$2,000",
        "q_lex": [
            "What is the bonus amount in the Referral Bonus Policy?",
            "How much does the Referral Bonus Policy pay for a successful referral?",
        ],
        "q_para": [
            "How large is the recruiting tip reward?",
            "How much do I get paid for bringing in a friend who gets hired?",
        ],
        "q2": "How long must a referred candidate stay employed before the bonus is paid?",
        "q_current": "What is the current bonus amount in the Referral Bonus Policy?",
        "q_unans": "Can contractors receive a bonus under the Referral Bonus Policy?",
    },
    {
        "id": "relocation_assistance",
        "name": "Relocation Assistance",
        "alias": "moving costs help",
        "sentence": "Approved relocations are reimbursed up to {v1}.",
        "v1": "$10,000",
        "q_lex": [
            "What is the reimbursement limit in the Relocation Assistance Policy?",
            "How much does the Relocation Assistance Policy reimburse at most?",
        ],
        "q_para": [
            "How much moving costs help is available?",
            "What is the most the firm will repay if I move cities for the job?",
        ],
        "q_unans": "Which moving companies are on the preferred list under the Relocation Assistance Policy?",
    },
    {
        "id": "tuition_reimbursement",
        "name": "Tuition Reimbursement",
        "alias": "school fee payback",
        "sentence": "The company reimburses up to {v1} per year in tuition for approved degree programs.",
        "v1": "$5,250",
        "v1_old": "$4,000",
        "q_lex": [
            "What is the yearly limit in the Tuition Reimbursement Policy?",
            "How much does the Tuition Reimbursement Policy cover per year?",
        ],
        "q_para": [
            "How much school fee payback can I claim in a year?",
            "What is the most the firm repays annually for my university classes?",
        ],
        "q_current": "What is the current yearly limit in the Tuition Reimbursement Policy?",
        "q_unans": "Which grades are required to qualify under the Tuition Reimbursement Policy?",
    },
    {
        "id": "bereavement_leave",
        "name": "Bereavement Leave",
        "alias": "time off after a family death",
        "sentence": "Employees may take {v1} of paid leave after the death of an immediate family member.",
        "v1": "5 days",
        "q_lex": [
            "How long is paid leave under the Bereavement Leave Policy?",
            "What is the length of leave in the Bereavement Leave Policy?",
        ],
        "q_para": [
            "How much time off after a family death is allowed?",
            "How many days off with pay do I get when a close relative passes away?",
        ],
        "q_unans": "Is Bereavement Leave available after the loss of a pet?",
    },
]

# Questions with no connection to any policy in the corpus.
OFFTOPIC_QUESTIONS: list[str] = [
    "What time does the cafeteria open?",
    "Where can I park a bicycle at the main office?",
    "Which floor is the finance team on?",
    "What is the dress code on Fridays?",
    "How do I book a meeting room?",
    "Who is the chief executive of Northwind Logistics?",
    "What is the guest Wi-Fi network called?",
    "Which public holidays does the company observe?",
    "What is the dental plan deductible?",
    "How many vacation days do new hires get?",
    "What is the company stock ticker symbol?",
    "Where is the nearest warehouse to the head office?",
    "How do I join the running club?",
    "What is the lunch-and-learn schedule this month?",
    "Does the office have a nursing room?",
    "What is the name of the employee shuttle service?",
]

FILLERS: list[str] = [
    "Managers apply this policy consistently across all teams.",
    "Exceptions require written approval from the People Team.",
    "This policy applies to all full-time employees.",
    "Part-time employees should ask their manager how it applies to them.",
    "The People Team reviews this policy each year.",
]
