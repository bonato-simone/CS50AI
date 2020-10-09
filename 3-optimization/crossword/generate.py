import sys

from crossword import *
from copy import deepcopy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.crossword.variables:
            self.domains[var] = set([value for value in self.domains[var] if var.length == len(value)])

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False
        x_index, y_index = overlap

        res = False
        tmp_domain = set()
        for x_value in self.domains[x]:
            if any(x_value[x_index] == y_value[y_index] for y_value in self.domains[y]):
                tmp_domain.add(x_value)
            else:
                res = True
        self.domains[x] = tmp_domain

        return res

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            for x in self.crossword.variables:
                for y in self.crossword.variables:
                    if x is not y:
                        arcs.append((x, y))
        while len(arcs):
            x, y = arcs.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor is not y and (neighbor, x) not in arcs:
                        arcs.append((neighbor, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if all(var in assignment for var in self.crossword.variables):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        if len(assignment) != len(set(assignment.values())):
            return False
        for x in assignment:
            if x.length != len(assignment[x]):
                return False
            for y in assignment:
                if x is not y:
                    overlap = self.crossword.overlaps[x, y]
                    if overlap is None:
                        continue
                    x_index, y_index = overlap
                    if assignment[x][x_index] != assignment[y][y_index]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        domains = list(self.domains[var])
        ruled_out_values_counters = []

        # for each value calculates the number of ruled out values
        for value in domains:
            ruled_out_values = 0
            for neighbor in self.crossword.neighbors(var) - set(assignment.keys()):
                if value in self.domains[neighbor]:
                    ruled_out_values += 1
                for neighbor_value in self.domains[neighbor] - {value}:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap is not None:
                        index, neighbor_index = overlap
                        if value[index] != neighbor_value[neighbor_index]:
                            ruled_out_values += 1
            ruled_out_values_counters.append(ruled_out_values)

        # zips domains' list and counters' list, orders by the latter and return elements from the former
        return [x for _, x in sorted(zip(ruled_out_values_counters, domains), reverse=True)]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = list(set(self.crossword.variables - assignment.keys()))

        # selects variables with fewer possible values
        min_values = min([len(self.domains[var]) for var in unassigned_variables])
        selected_variables = [var for var in unassigned_variables if len(self.domains[var]) == min_values]

        # if more than one variable is selected, among them selects variables with fewer neighbors
        if len(selected_variables) != 1:
            max_neighbors = max([len(self.crossword.neighbors(var)) for var in selected_variables])
            selected_variables = [var for var in selected_variables
                                  if len(self.crossword.neighbors(var)) == max_neighbors]
        return selected_variables[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value

            # checks if the new assignment is valid/consistent
            if self.consistent(assignment):

                # narrows the domain to the assigned value
                tmp_domains = deepcopy(self.domains)
                self.domains[var] = {value}
                arcs = [(neighbor, assigned_var) for assigned_var in assignment
                        for neighbor in self.crossword.neighbors(assigned_var)]
                self.ac3(arcs)

                res = self.backtrack(assignment)
                if res is not None:
                    return res

                # if a solution hasn't been found, restores the domains
                self.domains = tmp_domains

        del assignment[var]
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
