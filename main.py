import numpy as np
import plotly.graph_objs as go
from collections import deque


class Tower:
    """
    Class Tower, with square range, coordinates and neighborhoods.
    """
    def __init__(self, tower_range, value, signal, x, y):
        """
        :param tower_range: value of square range
        :param value: value tower on city map
        :param signal: tower signal for square range
        :param x: x coordinate tower
        :param y: y coordinate tower
        """
        self.range = tower_range
        self.value = value
        self.signal = signal
        self.x = x
        self.y = y
        # list of neighborhoods tower
        self.neighborhood = []

    def __repr__(self):
        """

        :return:
        """
        return f'{self.x}-{self.y}'


class GridCity:
    """
    Class city, where we must build tower and cover area by signal.
    Run run_experiment for test tasks.
    """
    def __init__(self, rows, cols, tower_r, observ_p, density=6):
        """
        :param rows: number of rows city grid
        :param cols: number of columns city grid
        :param tower_r: default range tower
        :param observ_p: probability of obstructed blocks
        :param density: density allocate towers for power influence of signal on non-tower blocks city
        """
        self.rows = rows
        self.cols = cols
        self.tower_r = tower_r
        self.observ_p = observ_p
        self.density = density
        self.budget = None
        self.cost_tower = None
        self.towers_available = []

        self.value = self.rows * self.cols // 2
        self.signal = 1

        _b = int(self.rows * self.cols * self.observ_p)
        self.city = np.concatenate([np.ones((_b,)) * self.rows * self.cols, np.zeros((self.rows * self.cols - _b,))])
        self.coverage = np.zeros((self.rows, self.cols))
        np.random.shuffle(self.city)
        self.city = np.reshape(self.city, (self.rows, self.cols))
        self.influence = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.towers_coordinates = []

    def set_budget_and_cost_tower(self, budget, cost=1):
        """
        Set of budget and cost tower. We can build towers only within budget.
        :param budget:
        :param cost:
        :return:
        """
        self.budget = budget
        self.cost_tower = cost

    def set_different_towers(self, towers_cost, towers_range):
        """
        We can set different cost and range of towers.
        :param towers_cost: list of costs
        :param towers_range: list of ranges
        :return:
        """
        if len(towers_cost) != len(towers_range):
            return 'Error. Length of list cost and range towers non equal.'
        else:
            self.towers_available = sorted(list(zip(towers_cost, towers_range)), key=lambda x: x[0])

    def build_tower(self, x, y, tower):
        """
        Build tower in city on concrete coordinates. And add coordinate on list of coordinates.
        :param x: x coordinate
        :param y: y coordinate
        :param tower: class tower
        :return:
        """
        self.city[y, x] += tower.value
        self.towers_coordinates.append([x, y])

    def activate_tower(self, x, y, tower):
        """
        Activate tower. It means that tower broadcast own signal on square area of range.
        :param x: x coordinate tower
        :param y: y coordinate tower
        :param tower: class tower
        :return:
        """
        self.coverage[max(0, y - tower.range): min(self.city.shape[0], y + tower.range + 1),
                      max(0, x - tower.range): min(self.city.shape[1], x + tower.range + 1)] += tower.signal
        for y_i in range(max(0, y - tower.range), min(y + tower.range + 1, self.rows)):
            for x_i in range(max(0, x - tower.range), min(x + tower.range + 1, self.cols)):
                self.influence[y_i][x_i].append(tower)

    def _find_start(self):
        """
        Find of coordinates for build first tower.
        :return: x and y coordinates, range and cost first tower.
        """
        range_tower = self.tower_r
        cost_tower = self.cost_tower
        x_start = self.tower_r
        y_start = self.tower_r
        if self.towers_available:
            max_cost_per_cell = 0
            for tower in self.towers_available:
                cost_per_cell = ((tower[1] * 2 + 1) ** 2) / tower[0]
                if cost_per_cell > max_cost_per_cell:
                    max_cost_per_cell = cost_per_cell
                    x_start = tower[1]
                    y_start = tower[1]
                    range_tower = tower[1]
                    cost_tower = tower[0]
        while x_start or y_start:
            if self.city[y_start, x_start] != self.rows * self.cols:
                break
            if x_start < y_start:
                y_start = max(0, y_start - 1)
            else:
                x_start = max(0, x_start - 1)
        return x_start, y_start, range_tower, cost_tower

    def _find_free_place(self, x, y, down, left, slice_down=False):
        """
        Find of coordinates where we can build tower. It means that coordinates do not according to obstructed block.
        :param x: start x coordinate for find of free block
        :param y: start y coordinate for find of free block
        :param down: flag who define can we make step down or up
        :param left: flag who define can we make step left or right
        :param slice_down: clarification of initial coordinates
        :return:
        If coverage on [x, y] block more than self.density, return None.
        Otherwise we return coordinate, value, range and cost tower to allocate.
        """
        if slice_down:
            while ((y - self.tower_r >= 0) and 
                   (0 in self.coverage[max(0, y - self.tower_r), 
                                       max(0, x - self.tower_r): min(x + self.tower_r + 1, self.cols - 1)])):
                y -= 1
            while ((y + self.tower_r < self.rows) and 
                   (0 not in self.coverage[min(y - self.tower_r + 1, self.rows - 1), 
                                           max(0, x - self.tower_r): min(x + self.tower_r + 1, self.cols - 1)])):
                y += 1
        if self.coverage[y, x] < self.density:
            if self.city[y, x] == self.rows * self.cols:

                while ((self.city[y, x] == self.rows * self.cols) and
                       (x >= 0 and y >= 0) and
                       (x < self.cols and y < self.rows)):
                    if down:
                        y -= 1
                        if self.city[y, x] == 0:
                            break
                        if left:
                            x -= 1
                        else:
                            x += 1
                    else:
                        y += 1
                        if self.city[y, x] == 0:
                            break
                        if left:
                            x -= 1
                        else:
                            x += 1

            if self.budget and (self.cost_tower or self.towers_available):
                if self.towers_available:
                    max_cost_per_cell = 0
                    cost_per_cell = 0
                    range_greatest_tower = self.tower_r
                    cost_greatest_tower = 0

                    for cost_tower, range_tower in self.towers_available:
                        if cost_tower <= self.budget:
                            cost_per_cell = np.count_nonzero(
                                self.coverage[max(0, y - range_tower): min(y + range_tower + 1, self.rows),
                                              max(0, x - range_tower): min(x + range_tower + 1, self.cols)] == 0)
                            cost_per_cell /= cost_tower
                            if cost_per_cell >= max_cost_per_cell:
                                range_greatest_tower = range_tower
                                cost_greatest_tower = cost_tower
                                max_cost_per_cell = cost_per_cell

                    return x, y, cost_per_cell, range_greatest_tower, cost_greatest_tower
                else:
                    additional_coverage = np.count_nonzero(
                        self.coverage[max(0, y - self.tower_r): min(y + self.tower_r + 1, self.rows),
                                      max(0, x - self.tower_r): min(x + self.tower_r + 1, self.cols)] == 0)
                    return x, y, additional_coverage, self.tower_r, self.cost_tower
            else:
                return x, y, 0, self.tower_r, 0

        else:
            return None

    def _zero_budget(self):
        """
        Check budget. Return True if we cannot build no more one of Tower and False otherwise.
        :return:
        """
        if self.towers_available:
            for cost, _ in self.towers_available:
                if self.budget >= cost:
                    return False
            return True
        else:
            if self.budget is not None and self.budget <= 0:
                return True
        return False

    def replacement_towers(self, optimize_hops=False):
        """
        Method who define how we search coordinate for build tower.
        We build first tower, define her border and try build next tower on korner of square range.
        We continue process till we can: till have budget or till do not cover all city are.
        :param optimize_hops: define step for build towers.
        If True - we optimize process of build towers for availability of signal exchange.
        :return:
        """
        x, y, range_tower, cost_tower = self._find_start()
        border_coverage = deque([[x, y, 0, range_tower, cost_tower]])

        while border_coverage:
            x, y, _, range_tower, cost_tower = border_coverage.pop()
            if self.budget and (self.cost_tower or self.towers_available):
                if cost_tower > self.budget:
                    continue
            if ((np.min(self.coverage[max(0, y - range_tower): min(y + range_tower + 1, self.rows),
                                      max(0, x - range_tower): min(x + range_tower + 1, self.cols)]) < self.density) and
                    (self.city[y, x] == 0)):
                tower = Tower(range_tower, self.value, self.signal, x, y)
                self.build_tower(x, y, tower)
                self.activate_tower(x, y, tower)

                if self.budget and (self.cost_tower or self.towers_available):
                    self.budget -= cost_tower
                    if self._zero_budget():
                        break
                    if self.towers_available:
                        while self.budget < self.towers_available[-1][0]:
                            self.towers_available.pop()
                            if not self.towers_available:
                                break
            else:
                continue

            if optimize_hops:
                next_tower = self._find_free_place(min(max(range_tower // 2, x + range_tower), self.cols - 1),
                                                   min(max(range_tower // 2, y + range_tower), self.rows - 1),
                                                   down=True, left=True)
                if next_tower:
                    border_coverage.append(next_tower)
                next_tower = self._find_free_place(min(max(range_tower // 2, x - range_tower), self.cols - 1),
                                                   min(max(range_tower // 2, y - range_tower), self.rows - 1),
                                                   down=False, left=False)
                if next_tower:
                    border_coverage.append(next_tower)
                next_tower = self._find_free_place(min(max(range_tower // 2, x + range_tower), self.cols - 1),
                                                   min(max(range_tower // 2, y - range_tower), self.rows - 1),
                                                   down=False, left=True)
                if next_tower:
                    border_coverage.append(next_tower)
                next_tower = self._find_free_place(min(max(range_tower // 2, x - range_tower), self.cols - 1),
                                                   min(max(range_tower // 2, y + range_tower), self.rows - 1),
                                                   down=True, left=False)
                if next_tower:
                    border_coverage.append(next_tower)
            else:
                next_tower = self._find_free_place(min(max(range_tower, x + range_tower * 2 - 2), self.cols - 1),
                                                   min(max(range_tower, y + 1), self.rows - 1),
                                                   down=True, left=True, slice_down=True)
                if next_tower:
                    border_coverage.append(next_tower)
                if x + range_tower >= self.cols:
                    x = range_tower
                    next_tower = self._find_free_place(min(max(range_tower, x), self.cols - 1),
                                                       min(max(range_tower, y + range_tower - 1), self.rows - 1),
                                                       down=True, left=True, slice_down=True)
                    if next_tower:
                        border_coverage.append(next_tower)

            if self.budget and (self.cost_tower or self.towers_available):
                border_coverage = sorted(border_coverage, key=lambda key_sort: key_sort[2])

    def build_network(self):
        """
        Connects towers in common network. Assign neighbors for each tower.
        :return:
        """
        for x, y in self.towers_coordinates:
            for main_tower in self.influence[y][x]:
                for tower_neighborhood in self.influence[y][x]:
                    if ((main_tower is not tower_neighborhood) and
                            (tower_neighborhood not in main_tower.neighborhood) and
                            (tower_neighborhood in self.influence[main_tower.y][main_tower.x])):
                        main_tower.neighborhood.append(tower_neighborhood)

    def get_path(self, source, target):
        """
        Find path of signal from source to target. Use BFS.
        :param source: from tower
        :param target: to tower
        :return:
        """
        x_source, y_source = self.towers_coordinates[source]
        x_target, y_target = self.towers_coordinates[target]

        source_tower = [t for t in self.influence[y_source][x_source] if (t.x == x_source and t.y == y_source)][0]
        target_tower = [t for t in self.influence[y_target][x_target] if (t.x == x_target and t.y == y_target)][0]
        neighborhood_source = deque([source_tower])
        visited = []
        path = {source_tower: [source_tower]}

        while neighborhood_source:
            neigh = neighborhood_source.popleft()

            if neigh in visited:
                continue
            else:
                visited.append(neigh)
                if target_tower is not neigh:
                    for t_n in neigh.neighborhood:
                        if t_n not in visited:
                            neighborhood_source.append(t_n)
                            if t_n in path:
                                if len(path[neigh] + [t_n]) < len(path[t_n]):
                                    path[t_n] = path[neigh] + [t_n]
                            else:
                                path[t_n] = path[neigh] + [t_n]
                else:
                    break
        return path[neigh]

    def visualize(self, path_signal=False, coverage=False, save_image=''):
        """
        Visualize result of grid of city. Can turn on/off coverage and path. Can save image to file or show on display.
        :param path_signal: if have - draw path signal, must be list of coordinates towers
        :param coverage: if True - draw coverage
        :param save_image: if have - save result to file image. must be string
        :return:
        """
        data = []
        if coverage:
            data.append(go.Heatmap(z=self.coverage,
                                   y=list(range(self.rows)),
                                   x=list(range(self.cols)),
                                   colorscale=[[0., 'white'],
                                               [.5, 'green'],
                                               [1., 'orange']],
                                   zmin=0,
                                   zauto=False,
                                   showscale=True))
        data.append(go.Heatmap(z=self.city,
                               y=list(range(self.rows)),
                               x=list(range(self.cols)),
                               colorscale=[[0, 'rgba(200, 200, 200, 0)'],
                                           [1. / 2, 'red'],
                                           [1., 'rgba(50, 50, 50, .8)']] if coverage else [
                                   [0, 'rgba(200, 200, 200, 1)'],
                                   [1. / 2, 'red'],
                                   [1., 'rgba(50, 50, 50, .8)']],
                               showscale=False,
                               hoverinfo='skip' if coverage else 'x+y'
                               ))
        if path_signal:
            xs = []
            ys = []
            for t in path_signal:
                xs.append(t.x)
                ys.append(t.y)
            data.append(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#0000ff', width=5)))

        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(xaxis_scaleanchor="y",
                          height=1000,
                          width=1000,
                          showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        if save_image:
            fig.write_image(save_image)
        else:
            fig.show()

    def run_experiment(self, task_number, image_path=''):
        """
        Run experiments according to task number.
        task 0 - visualize only city grid
        task 1 - visualize replacement tower and coverage without pathway of signal
        task 2 - visualize replacement tower and coverage with pathway of signal
        task 3 - when we have budget and cost towers
        task 4 - when we have budget and different cost and range towers
        :param task_number: int
        :param image_path: string
        :return:
        """
        if task_number == 0:
            self.visualize(save_image=image_path)
        if task_number == 1:
            self.replacement_towers()

            print(f'FULL COVERAGE IS: {self.coverage.all()}')

            self.visualize(coverage=True, save_image=image_path)
        if task_number == 2:
            self.replacement_towers(optimize_hops=True)
            self.build_network()

            print(f'FULL COVERAGE IS: {self.coverage.all()}')

            path_between_towers = self.get_path(np.random.randint(len(self.towers_coordinates)),
                                                np.random.randint(len(self.towers_coordinates)))
            self.visualize(path_signal=path_between_towers, coverage=True, save_image=image_path)
        if task_number == 3:
            self.set_budget_and_cost_tower(np.random.randint(1000, 30000),
                                           np.random.randint(10, 100))
            self.replacement_towers(optimize_hops=True)
            self.build_network()

            print(f'FULL COVERAGE IS: {self.coverage.all()}')
            print(f'BUDGET REMAINING: {self.budget}')

            path_between_towers = self.get_path(np.random.randint(len(self.towers_coordinates)),
                                                np.random.randint(len(self.towers_coordinates)))
            self.visualize(path_signal=path_between_towers, coverage=True, save_image=image_path)
        if task_number == 4:
            self.set_budget_and_cost_tower(np.random.randint(1000, 30000))
            nums_towers = np.random.randint(3, 8)
            self.set_different_towers(np.random.randint(10, 1000, size=nums_towers),
                                      np.random.randint(10, 100, size=nums_towers))
            self.replacement_towers(optimize_hops=True)
            self.build_network()

            print(f'FULL COVERAGE IS: {self.coverage.all()}')
            print(f'BUDGET REMAINING: {self.budget}')

            path_between_towers = self.get_path(np.random.randint(len(self.towers_coordinates)),
                                                np.random.randint(len(self.towers_coordinates)))
            self.visualize(path_signal=path_between_towers, coverage=True, save_image=image_path)


if __name__ == '__main__':
    for task in range(5):
        for test_number in range(3):
            print(f'TASK {task}, TEST {test_number}')
            gc = GridCity(np.random.randint(100, 600), np.random.randint(100, 600), np.random.randint(5, 20),
                          np.random.randint(30, 50) / 100, 4)
            gc.run_experiment(task_number=task, image_path=f'task{task}_test{test_number}.png')
