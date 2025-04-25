{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15cab5e4-15a7-4fb9-965d-63314faf047d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "\n",
    "class HillClimbing:\n",
    "    def __init__(self, initial_plan, cost_function, generate_neighbors, max_iters=1000, delta_fraction=0.01):\n",
    "        # Current best solution and its score\n",
    "        self.current = initial_plan\n",
    "        self.score = cost_function(initial_plan)\n",
    "\n",
    "        # Store references for use during search\n",
    "        self.cost_function = cost_function\n",
    "        self.generate_neighbors = generate_neighbors\n",
    "        self.max_iters = max_iters\n",
    "        self.delta_fraction = delta_fraction\n",
    "\n",
    "    def run(self):\n",
    "        for iteration in range(self.max_iters):\n",
    "            delta = self.delta_fraction * self.current['total_salary']\n",
    "            neighbors = self.generate_neighbors(self.current, delta)\n",
    "            best_neighbor = None\n",
    "            best_score = self.score\n",
    "            for neighbor in neighbors:\n",
    "                s = self.cost_function(neighbor)\n",
    "                if s > best_score:\n",
    "                    best_score = s\n",
    "                    best_neighbor = neighbor\n",
    "            if best_neighbor:\n",
    "                self.current = best_neighbor\n",
    "                self.score = best_score\n",
    "            else:\n",
    "                break\n",
    "        return self.current, self.score"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
