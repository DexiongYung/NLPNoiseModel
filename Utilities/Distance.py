import torch
import math


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_levenshtein_w_counts(s1: str, s2: str):
    row_dim = len(s1) + 1  # +1 for empty string
    height_dim = len(s2) + 1

    # tuple = [ins, del, subs]
    # Moving across row is insertion
    # Moving down column is deletion
    # Moving diagonal is sub
    matrix = [[[n, 0, 0] for n in range(row_dim)] for m in range(height_dim)]

    for i in range(1, height_dim):
        matrix[i][0][1] = i

    for y in range(1, height_dim):
        for x in range(1, row_dim):
            left_scores = matrix[y][x - 1].copy()
            above_scores = matrix[y - 1][x].copy()
            diagonal_scores = matrix[y - 1][x - 1].copy()

            scores = [sum_list(left_scores), sum_list(
                diagonal_scores), sum_list(above_scores)]
            min_idx = scores.index(min(scores))

            if min_idx == 0:
                matrix[y][x] = left_scores
                matrix[y][x][0] += 1
            elif min_idx == 1:
                matrix[y][x] = diagonal_scores
                matrix[y][x][2] += (s1[x-1] != s2[y-1])
            else:
                matrix[y][x] = above_scores
                matrix[y][x][1] += 1

    result = matrix[-1][-1]
    # 0 is insertions, 1 is removals and 2 is substitutions
    return result[0], result[1], result[2]


def get_indexes_of_edits(s1: str, s2: str):
    row_dim = len(s1) + 1  # +1 for empty string
    height_dim = len(s2) + 1

    # tuple = [ins, del, subs]
    # Moving across row is insertion
    # Moving down column is deletion
    # Moving diagonal is sub
    matrix = [[n for n in range(row_dim)] for m in range(height_dim)]

    for i in range(1, height_dim):
        matrix[i][0] = i

    for y in range(1, height_dim):
        for x in range(1, row_dim):
            left_score = matrix[y][x - 1]
            above_score = matrix[y - 1][x]
            diagonal_score = matrix[y - 1][x - 1]

            scores = [left_score, diagonal_score, above_score]
            min_idx = scores.index(min(scores))

            if min_idx == 0:
                matrix[y][x] = left_score + 1
            elif min_idx == 1:
                matrix[y][x] = diagonal_score + (s1[x-1] != s2[y-1])
            else:
                matrix[y][x] = above_score + 1

    x, y = 0, 0
    edit_idxes = []
    prev_score = 0
    row_max = row_dim - 1
    column_max = height_dim - 1
    while x < row_max or y < column_max:
        is_l_x_max = x < row_max 
        is_l_y_max = y < column_max
        right_score = matrix[y][x+1] if is_l_x_max else float('inf')
        below_score = matrix[y+1][x] if is_l_y_max else float('inf')
        diagonal_score = matrix[y+1][x+1] if is_l_x_max and is_l_y_max else float('inf')

        scores = [right_score, diagonal_score, below_score]
        min_score = min(scores)
        min_score_idx = scores.index(min_score)

        if min_score_idx == 0:
            x += 1
        elif min_score_idx == 1:
            x += 1
            y += 1
        else:
            y += 1
        
        if prev_score < matrix[y][x]:
            edit_idxes.append(x - 1)
        
        prev_score = min_score
        
    return edit_idxes


def sum_list(lst: list):
    sum = 0

    for i in lst:
        sum += i

    return sum


def get_levenshtein_winner(noised_names: list, name: str):
    distance = math.inf
    winner = ''
    for noised_name in noised_names:
        curr_distance = levenshtein(noised_name, name)

        if curr_distance < distance and curr_distance != 0:
            distance = curr_distance
            winner = noised_name
    return winner
