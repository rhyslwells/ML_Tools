def rank_name_generator(name):
    """Expands single string which is a Dota 2 rank to list of rank steps.

       Parameters:
       -----------
       name: string
            Name of the rank.

       Returns:
       -----------
       ranks: list
            List of 7 different rank steps attached to rank name.
    """
    roman_numbers = ["I", "II", "III", "IV", "V", "VI", "VII"]
    ranks = ["{} {}".format(name, n) for n in roman_numbers]
    return ranks
