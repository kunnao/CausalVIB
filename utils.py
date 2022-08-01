# Removes the spaces from a string. Used for parsing terminal inputs.
def remove_space(string):
    return string.replace(" ", "")


def str2int_list(string, sep=','):
    if string is None:
        return None
    try:
        return [int(x) for x in string.split(sep)]
    except Exception as e:
        print(e)
    try:
        x = [int(x) for x in string.split('-')]
        return [i for i in range(x[0], x[1])]
    except Exception as e:
        print(e)
    finally:
        return None
