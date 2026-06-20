import os

from app.llm.client import LlmClient
from app.llm.skill_router import SkillRouter
from dotenv import load_dotenv

load_dotenv()

llm_client = LlmClient(
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
)

skill_router = SkillRouter(llm_client.chat_openai)

messages_skills : list[tuple[str, str]] = [
    # Q&A — clear cases
    ("Which laptop do you recommend for gaming?",            "qa"),
    ("How much does a mouse cost?",                          "qa"),
    ("What are the specs of the RTX 4070?",                  "qa"),
    ("Do you have wireless headphones?",                     "qa"),
    ("What is the difference between SSD and HDD?",          "qa"),
    ("Which monitor is best for photo editing?",             "qa"),
    ("How much RAM does this laptop have?",                  "qa"),
    ("Is the Sony WH-1000XM5 worth buying?",                 "qa"),
    ("What keyboards do you have in stock?",                 "qa"),
    ("Tell me about the MacBook Pro",                        "qa"),
    ("What is the battery life on this laptop?",             "qa"),
    ("Do you sell gaming chairs?",                           "qa"),
    ("What is the price range for your monitors?",           "qa"),
    ("Compare the two headphone models",                     "qa"),
    ("Does this mouse have RGB lighting?",                   "qa"),
    ("Which CPU is faster, i7 or i9?",                       "qa"),
    ("What is included in the laptop package?",              "qa"),

    # Order — clear cases
    ("I want to buy headphones",                             "order"),
    ("Order me a monitor",                                   "order"),
    ("I would like to purchase a keyboard",                  "order"),
    ("Place an order for a gaming mouse",                    "order"),
    ("Buy me the cheapest laptop",                           "order"),
    ("I want to cancel my order",                            "order"),
    ("I want to return my order",                            "order"),
    ("I want to buy 2 monitors",                             "order"),
    ("Can you order the RTX 4070 for me?",                   "order"),
    ("I need to cancel my purchase",                         "order"),
    ("Process a refund for my order",                        "order"),
    ("I would like to exchange my keyboard",                 "order"),
    ("Return the headphones I bought",                       "order"),
    ("Remove my order",                                      "order"),
    ("I want to get the wireless mouse",                     "order"),
    ("I want to cancel order #123",                          "order"),
    ("Add a mouse to my cart",                               "order"),

    # Track — clear cases
    ("Where is my package?",                                 "track"),
    ("When will order #456 arrive?",                         "track"),
    ("Order status",                                         "track"),
    ("What is the status of my order?",                      "track"),
    ("Has my order shipped yet?",                            "track"),
    ("Where is my delivery?",                                "track"),
    ("Track my order",                                       "track"),
    ("Is my package on the way?",                            "track"),
    ("My order has not arrived yet",                         "track"),
    ("Check my shipment status",                             "track"),
    ("When will my monitor be delivered?",                   "track"),
    ("Track shipment #789",                                  "track"),
    ("What is the ETA for my order?",                        "track"),

    # Edge cases — easy to confuse
    ("I want to return the item, what is the status?",       "order"),
    ("Cancel and refund my order",                           "order"),
    ("Where is the laptop I ordered?",                       "track"),
    ("Did my order go through?",                             "track"),
    ("I am not happy with my order, I want a refund",        "order"),
    ("My package is late, can you check?",                   "track"),
    ("I changed my mind, cancel everything",                 "order"),
]

async def evaluate():

    wrong_predictions: dict[str, int] = {}

    correct = 0
    for message, expected_skill in messages_skills:
        result = await skill_router.classify(message)
        result_correct = result.skill == expected_skill
        if result_correct:
            correct += 1
        else:
            key = expected_skill + ":" + result.skill
            wrong_predictions[key] = wrong_predictions.get(key, 0) + 1

        print(f"Message: \"{message}\", expected: {expected_skill}, router result: {result.skill} - { "CORRECT" if result_correct else "WRONG" }")

    print(f"Accuracy: {correct / len(messages_skills):.2%}")

    print(f"Wrong predictions:")
    for key, count in wrong_predictions.items():
        expected_skill, result_skill = key.split(":")
        print(f"  expected={expected_skill}, got:{result_skill} x {count}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate())