import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return set(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def get_neighboring_cells(self, cell):
        """
        Returns set of cells adjacent to a cell.
        """
        neighboring_cells = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if 0 <= i < self.height and 0 <= j < self.width and (i != cell[0] or j != cell[1]):
                    neighboring_cells.add((i, j))
        return neighboring_cells

    def add_sentence(self, cells, count):
        """
        Adds a sentence to knowledge and updates mines and safe cells.
        Returns True if changes have been made, False otherwise.
        """
        return_value = False

        new_sentence = Sentence(cells, count)
        self.knowledge.append(new_sentence)

        cell_has_been_marked = True
        while cell_has_been_marked:
            cell_has_been_marked = False
            knowledge_tmp = []
            cells_to_mark_mine = []
            cells_to_mark_safe = []

            for s in self.knowledge:
                known_mines = s.known_mines()
                known_safes = s.known_safes()

                if len(known_mines) or len(known_safes):
                    for cell in known_mines:
                        cells_to_mark_mine.append(cell)
                    for cell in known_safes:
                        cells_to_mark_safe.append(cell)
                    cell_has_been_marked = return_value = True
                elif s not in knowledge_tmp and len(s.cells):
                    knowledge_tmp.append(s)

            self.knowledge = knowledge_tmp
            for cell in cells_to_mark_mine:
                self.mark_mine(cell)
            for cell in cells_to_mark_safe:
                self.mark_safe(cell)

        return return_value

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)
        self.add_sentence({cell}, 0)

        # selects neighboring cells and removes mines and safe cells from them
        neighboring_cells = self.get_neighboring_cells(cell)
        neighboring_cells_clean = set()

        for cell in neighboring_cells:
            if cell in self.mines:
                count -= 1
            elif cell not in self.safes:
                neighboring_cells_clean.add(cell)

        self.add_sentence(neighboring_cells_clean, count)
        sentence_has_been_added = True

        # tries to find new sentences
        while sentence_has_been_added:
            sentences_to_add = []
            sentence_has_been_added = False

            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 is not s2:
                        cells1, count1, cells2, count2 = s1.cells, s1.count, s2.cells, s2.count
                        if cells1.issubset(cells2):
                            sentences_to_add.append((cells2 - cells1, count2 - count1))

            for s in sentences_to_add:
                if self.add_sentence(*s):
                    sentence_has_been_added = True

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for safe in self.safes:
            if safe not in self.moves_made:
                return safe
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for i in range(self.height):
            for j in range(self.width):
                cell = (i, j)
                if cell not in (self.moves_made | self.mines):
                    return cell
        return None
