import torch
import accuracy_utils
import ujson as json


def estimate_mean_and_error(Y, include_var_X=False):
    """
    Estimate the mean and margin of error for a 95% confidence interval

    Parameters:
    Y: PyTorch tensor of shape (N, M) containing the Y_{i,j} samples

    Returns:
    Y_hat: estimated mean
    margin_of_error: half-width of the 95% confidence interval
    """
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)

    Y = Y.float()
    N, M = Y.shape

    # Calculate mean estimator
    Y_hat = torch.mean(Y)
    if M == 1:
        return Y_hat.item(), 1.96 * (torch.var(Y).item() / N)**0.5


    # Step 1: Calculate sample means for each i
    mu_i_hat = torch.mean(Y, dim=1)  # Shape: (N,)

    # Step 2: Estimate conditional variances for each i
    sigma_i_squared_hat = torch.var(Y, dim=1, unbiased=True)  # Shape: (N,)

    # Step 3: Average of conditional variances
    E_sigma_i_squared_hat = torch.mean(sigma_i_squared_hat)

    if include_var_X:
        Var_mu_i_hat = torch.var(mu_i_hat, unbiased=True)
        Var_Y_hat = E_sigma_i_squared_hat / (N * M) + Var_mu_i_hat / N
    else:
        Var_Y_hat = E_sigma_i_squared_hat / (N * M)

    # Step 6: Margin of error using fixed z-score of 1.96 for 95% CI
    margin_of_error = 1.96 * torch.sqrt(Var_Y_hat)

    return Y_hat.item(), margin_of_error.item()


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def bon(rewards, n):
    # given rewards: [N, K], return the
    assert n > 0 and len(rewards) > 0
    assert len(rewards) % n == 0
    n_groups = len(rewards) // n
    bon_rewards = []
    for i in range(n_groups):
        group_rewards = rewards[i*n:(i+1)*n]
        bon_rewards.append(any(group_rewards))
    return bon_rewards


def classifier_bon(rewards, classifier_values, n):
    assert n > 0 and len(rewards) > 0 and len(rewards) == len(classifier_values)
    assert len(rewards) % n == 0
    n_groups = len(rewards) // n
    bon_rewards = []
    for i in range(n_groups):
        group_rewards = rewards[i*n:(i+1)*n]
        group_classifier_values = classifier_values[i*n:(i+1)*n]
        classifier_chosen_idx = torch.argmax(torch.tensor(group_classifier_values)).item()
        bon_rewards.append(group_rewards[classifier_chosen_idx])
    return bon_rewards


def weighted_majority(rewards, gen_answers, n, weights):
    assert n > 1 and len(gen_answers) == len(weights), f"len(gen_answers) = {len(gen_answers)}, len(weights) = {len(weights)}, n = {n}"
    assert len(gen_answers) % n == 0, f"len(gen_answers) = {len(gen_answers)}, n = {n}"
    n_groups = len(gen_answers) // n

    outputs = {}
    outputs['maj_rewards'] = []
    outputs['wmaj_rewards'] = []
    for i in range(n_groups):
        group_gen_answers = gen_answers[i*n:(i+1)*n]
        group_weights = weights[i*n:(i+1)*n]
        group_rewards = rewards[i*n:(i+1)*n]
        import functools
        equivalence_relation = functools.partial(accuracy_utils.math_verify_check)
        gen_answers_partition, weights_partition, rewards_partition = accuracy_utils.equivalence_partition_with_weights(
            group_gen_answers, group_weights, equivalence_relation, group_rewards)

        for name in ['maj_rewards', 'wmaj_rewards']:
            if name == 'maj_rewards':
                partition_scores = [len(weights) for weights in weights_partition]
            else:
                partition_scores = [sum(weights) for weights in weights_partition]
            max_score = max(partition_scores)
            accuracy = 0
            num_longest = 0
            for partition_idx in range(len(partition_scores)):
                if abs(partition_scores[partition_idx]-max_score) < 1e-5:
                    cur_rewards = rewards_partition[partition_idx]
                    assert all(cur_rewards) or not any(cur_rewards), cur_rewards
                    accuracy += cur_rewards[0]
                    num_longest += 1
            outputs[name].append(accuracy / num_longest)
    return outputs
