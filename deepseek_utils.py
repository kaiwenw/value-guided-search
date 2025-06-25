import re

def format_roll_in(prompt):
    return f"<｜begin▁of▁sentence｜><｜User｜>{prompt} Please think step-by-step and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"


def remove_thinking_text(s):
    return re.sub(r".*?</think>\n?\n?", "", s, flags=re.DOTALL)
