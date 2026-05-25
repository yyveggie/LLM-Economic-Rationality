"""Prompt templates for the four preference domains and three conditions.

These follow the SI of Chen, Liu, Shan & Zhong (2023):
  Domains   : risk / time / social / food
  Conditions: baseline / price_framing / discrete_choice
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from .tasks import BudgetTask


# -----------------------------------------------------------------------------
# Demographic injection (used by Fig.4 variants)
# -----------------------------------------------------------------------------
DEMOGRAPHIC_DESCRIPTORS: Dict[str, str] = {
    "none": "human",
    "female": "female",
    "male": "male",
    "kid": "young child",
    "elder": "elderly",
    "low_edu": "person with an elementary school education",
    "high_edu": "person with a college education",
    "asian": "Asian",
    "black": "African American",
}


def system_prompt(demographic: str = "none") -> str:
    descriptor = DEMOGRAPHIC_DESCRIPTORS.get(demographic, "human")
    return (
        f"I want you to act as a {descriptor} decision maker. You will be given "
        "25 rounds of decision-making tasks and will be responsible for making "
        "decisions. You should use your best judgment to come up with solutions "
        "that you like most. My first request is \"You must provide your answers "
        "in every round.\" If you do not provide an answer, I will assume that "
        "you make a random choice."
    )


# -----------------------------------------------------------------------------
# Per-domain assistant background message + per-task user message
# -----------------------------------------------------------------------------

def assistant_background(domain: str, condition: str) -> str:
    if domain == "risk":
        if condition in ("baseline", "price_framing"):
            return (
                "In every round, the decision maker has 100 points that need to "
                "be invested between Asset A and Asset B. The decision maker has "
                "a 50% chance to get the return from Asset A or the other 50% "
                "chance to get the return from Asset B. First please only tell "
                "me the number of points for investing Asset A, then please only "
                "tell me the number of points for investing Asset B."
            )
        if condition == "discrete_choice":
            return (
                "In every round, the decision maker will be presented with 11 "
                "options, each represented in the form ($M, $N). The decision "
                "maker has a 50% chance to get M from Asset A or the other 50% "
                "chance to get N from Asset B. Please only tell me about your "
                "best option in every round."
            )
    if domain == "time":
        if condition in ("baseline", "price_framing"):
            return (
                "In every round, the decision maker has 100 points that need to "
                "be invested between today and one month later. The decision "
                "maker will get dollars today from the points invested today and "
                "will get a check that can be cashed in one month later from the "
                "points invested one month later. Please first only tell me the "
                "number of points for investing today, then please only tell me "
                "the number of points for investing one month later."
            )
        if condition == "discrete_choice":
            return (
                "In every round, the decision maker will be presented with 11 "
                "options, each represented in the form ($M, $N). The decision "
                "maker will get $M today and a check of $N that can be cashed "
                "one month later. Please only tell me about your best option in "
                "every round."
            )
    if domain == "social":
        if condition in ("baseline", "price_framing"):
            return (
                "In every round, the decision maker is randomly matched with a "
                "new anonymous subject and there is no further interaction "
                "between them. The decision maker has 100 points that need to be "
                "allocated between him/herself and the other one. Please first "
                "only tell me the number of points for him/herself, then please "
                "only tell me the number of points for the other one."
            )
        if condition == "discrete_choice":
            return (
                "In every round, the decision maker is randomly matched with a "
                "new anonymous subject. The decision maker will be presented "
                "with 11 options, each represented in the form ($M, $N), where "
                "$M goes to him/herself and $N goes to the other one. Please "
                "only tell me about your best option in every round."
            )
    if domain == "food":
        if condition in ("baseline", "price_framing"):
            return (
                "In every round, the decision maker has 100 points that need to "
                "be spent between ham meat and tomato. Please first only tell me "
                "the number of points spent on ham meat, then please only tell "
                "me the number of points spent on tomato."
            )
        if condition == "discrete_choice":
            return (
                "In every round, the decision maker will be presented with 11 "
                "options, each represented in the form (M Kg, N Kg), where M is "
                "kilograms of ham meat and N is kilograms of tomato. Please only "
                "tell me about your best option in every round."
            )
    raise ValueError(f"Unknown domain/condition: {domain}/{condition}")


def task_message(domain: str, condition: str, task: BudgetTask) -> str:
    M, N = task.M, task.N

    if condition == "baseline":
        if domain == "risk":
            return (f"In this round, investing every 1 point for Asset A returns "
                    f"{M:.2f} dollars, and investing every 1 point for Asset B "
                    f"returns {N:.2f} dollars. What is your allocation?")
        if domain == "time":
            return (f"In this round, investing every 1 point for today returns "
                    f"{M:.2f} dollars today, and investing every 1 point for "
                    f"one month later returns {N:.2f} dollars check which can "
                    f"be cashed in one month later. What is your allocation?")
        if domain == "social":
            return (f"In this round, allocating every 1 point for him/herself "
                    f"returns {M:.2f} dollars, and allocating every 1 point for "
                    f"the other one returns {N:.2f} dollars for him/her. What "
                    f"is your allocation?")
        if domain == "food":
            return (f"In this round, every 1 point spent on ham meat will get "
                    f"{M:.2f} Kg ham meat, and every 1 point spent on tomato "
                    f"will get {N:.2f} Kg tomato. What is your allocation?")

    if condition == "price_framing":
        invM, invN = 1.0 / M, 1.0 / N
        if domain == "risk":
            return (f"In this round, investing every {invM:.2f} points for "
                    f"Asset A returns 1 dollar, and investing every {invN:.2f} "
                    f"points for Asset B returns 1 dollar. What is your "
                    f"allocation?")
        if domain == "time":
            return (f"In this round, investing every {invM:.2f} points for "
                    f"today returns 1 dollar today, and investing every "
                    f"{invN:.2f} points for one month later returns 1 dollar "
                    f"check which can be cashed in one month later. What is "
                    f"your allocation?")
        if domain == "social":
            return (f"In this round, allocating every {invM:.2f} points for "
                    f"him/herself returns 1 dollar, and allocating every "
                    f"{invN:.2f} points for the other one returns 1 dollar for "
                    f"him/her. What is your allocation?")
        if domain == "food":
            return (f"In this round, every {invM:.2f} points spent on ham meat "
                    f"will get 1 Kg ham meat, and every {invN:.2f} points spent "
                    f"on tomato will get 1 Kg tomato. What is your allocation?")

    if condition == "discrete_choice":
        # 11 options indexed i = 0..10, splitting the budget at 10*i points for A
        options = []
        for i in range(11):
            xA_pts = 10 * i        # points to A
            xB_pts = 100 - 10 * i  # points to B
            payoff_A = round(xA_pts * M, 2)
            payoff_B = round(xB_pts * N, 2)
            options.append(f"(${payoff_A}, ${payoff_B})")
        return ("In this round, there are 11 options, which are "
                + ", ".join(options) + ". Which is the best?")

    raise ValueError(f"Unknown condition: {condition}")


@dataclass
class ChatPrompt:
    system: str
    assistant: str
    user: str


def build_prompt(domain: str,
                 condition: str,
                 task: BudgetTask,
                 demographic: str = "none") -> ChatPrompt:
    return ChatPrompt(
        system=system_prompt(demographic),
        assistant=assistant_background(domain, condition),
        user=task_message(domain, condition, task),
    )
