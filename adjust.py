import difflib
from main import main


def adjust_value(from_: int, to_: int, step=None):
    expected_output = '123456789*0#874995*888285837**40#*9135*351#043387301#149951161#978567013353#'
    best_value = (0, 1000, '')  # (value, errors)
    for i in range(from_, to_):
        output = main(r'challenge 2022.wav', i)
        loss = len(list(difflib.ndiff(output, expected_output)))
        if loss <= best_value[1]:
            best_value = (i, loss, output)

    if step is None:
        print(best_value)
        return

    output_higher = main(r'challenge 2022.wav', best_value[0] + step)
    output_lower = main(r'challenge 2022.wav', best_value[0] - step)
    loss_higher = len(list(difflib.ndiff(expected_output, output_higher)))
    loss_lower = len(list(difflib.ndiff(expected_output, output_lower)))

    if loss_lower > best_value[1] and loss_higher > best_value[1]:
        print(best_value)
        return
    elif loss_lower < best_value[1]:
        multiply = -1
        new_loss = loss_lower
    elif loss_higher <= best_value[1]:
        multiply = 1
        new_loss = loss_higher
    else:
        raise ValueError

    while new_loss < best_value[1]:
        best_value = (best_value[0] + (step * multiply), new_loss)
        output = main(r'challenge 2022.wav', best_value[0] + (step * multiply))
        new_loss = len(list(difflib.ndiff(output, expected_output)))
    print(best_value)


if __name__ == "__main__":
    adjust_value(1, 70, 0.1)
