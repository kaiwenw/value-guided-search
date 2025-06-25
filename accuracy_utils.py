import itertools
import math_verify
from math_verify.utils import timeout
from math_verify.errors import TimeoutException
from sympy import Basic, MatrixBase
import re


# the following functions are math_verify that doesn't catch the exception
def compare_single_extraction(
    gold: Basic | MatrixBase | str,
    target: Basic | MatrixBase | str,
    float_rounding: int,
    numeric_precision: int,
    timeout_seconds: int,
    strict: bool = True,
) -> bool:

    @timeout(timeout_seconds=timeout_seconds)
    def wrapped_fn():
        nonlocal gold, target
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(
            target, (Basic, MatrixBase)
        ):
            return math_verify.grader.sympy_expr_eq(
                gold, target, float_rounding, numeric_precision, strict
            )

        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    try:
        return wrapped_fn()
    except Exception as e:
        return False

    except TimeoutException:
        return False


def math_verify_with_exception(parsed_x, parsed_y, numeric_precision, float_rounding, timeout_seconds):
    return any(
        compare_single_extraction(x, y, numeric_precision=numeric_precision, float_rounding=float_rounding, timeout_seconds=timeout_seconds)
        for x, y in itertools.product(parsed_x, parsed_y)
    )


def math_verify_check(expr1, expr2, symmetric=True, timeout_seconds=1, numeric_precision=5, float_rounding=5):
    if expr1 == "" or expr2 == "":
        return False

    parsed_expr1 = math_verify.parse(f"${expr1}$")
    parsed_expr2 = math_verify.parse(f"${expr2}$")
    result = math_verify_with_exception(
        parsed_expr1, parsed_expr2, numeric_precision=numeric_precision, float_rounding=float_rounding, timeout_seconds=timeout_seconds)
    if symmetric:
        reversed_result = math_verify_with_exception(
            parsed_expr2, parsed_expr1, numeric_precision=numeric_precision, float_rounding=float_rounding, timeout_seconds=timeout_seconds)
        return result or reversed_result
    else:
        return result


def equivalence_partition_with_weights(iterable, iterable_weights, relation, iterable_rewards):
    """Partitions a set of objects into equivalence classes.

  Note that this code assumes---but does not check---that the given
  relation is actually an equivalence relation, meaning that it is symmetric,
  reflexive, and transitive.

  Args:
      iterable: collection of objects to be partitioned
      relation: equivalence relation. I.e. relation(o1,o2) evaluates to True if
        and only if o1 and o2 are equivalent

  Returns:
      classes: A sequence of lists
  """
    classes = []
    class_weights = []
    class_rewards = []
    for obj, obj_weight, obj_reward in zip(iterable, iterable_weights, iterable_rewards):  # for each object
        # find the class it is in
        if obj is None:
            classes.append([obj])
            class_weights.append([obj_weight])
            class_rewards.append([obj_reward])
            continue
        found = False
        for class_index, cl in enumerate(classes):
            if not cl[0]:  # modification
                continue
            if relation(cl[0], obj):  # is it equivalent to this class?
                cl.append(obj)
                class_weights[class_index].append(obj_weight)
                class_rewards[class_index].append(obj_reward)
                found = True
                break
        if not found:  # it is in a new class
            classes.append([obj])
            class_weights.append([obj_weight])
            class_rewards.append([obj_reward])
    return classes, class_weights, class_rewards


def find_boxed_content(s, last_occurrence):
    pattern = r'\\boxed\{'
    all_matches = [m.end() for m in re.finditer(pattern, s)]
    if len(all_matches) == 0:
        return None
    if last_occurrence:
        start = all_matches[-1]
    else:
        start = all_matches[0]
    stack = 1
    i = start
    while i < len(s) and stack > 0:
        if s[i] == '{':
            stack += 1
        elif s[i] == '}':
            stack -= 1
        i += 1
    if stack == 0:
        return s[start:i - 1]  # Return the content inside the braces

    return None

def extract_between_and_with_boxes(
        x: str,
        last_occurrence: bool = False,
) -> str:
    """Extracts the boxed or delimited answer, returning empty string otherwise."""
    # deprecated extract_between
    # answer = extract_between(x, last_occurrence=last_occurrence)
    boxed_answer = find_boxed_content(x, last_occurrence=last_occurrence)
    if boxed_answer is not None:
        return boxed_answer
    else:
        return ''


def process_sample(sample, extract_last_occurrence=True):
    # few_shot_separator used to prevent model hallucinating new problems
    # extract last occurrence should be turned on in most cases
    if sample is None:
        return ''

    boxed_answer = find_boxed_content(sample, extract_last_occurrence)
    if boxed_answer is None:
        return ''

    return boxed_answer.strip()