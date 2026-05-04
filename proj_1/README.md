# Flappy Bird with Deep Reinforcement Learning (DQN)

Υλοποίηση agent για το παιχνίδι Flappy Bird με χρήση **Deep Q-Networks (DQN)** σε περιβάλλον custom OpenAI Gym–style.

---

## 🎯 Στόχος

Να εκπαιδευτεί ένας agent που μαθαίνει να ελέγχει το Flappy Bird (flap / no-op) μεγιστοποιώντας το συνολικό reward και το score (pipes passed).

---

## 🧠 Μέθοδος

* **State (4D):**

  * `bird_y`
  * `bird_velocity`
  * `pipe_x`
  * `bird_y - pipe_gap_y`

* **Actions:**

  * `0`: Do nothing
  * `1`: Flap

* **Model:**

  * Fully Connected Neural Network (MLP)
  * 4 → 64 → 64 → 2

* **Algorithm:**

  * Deep Q-Network (DQN)
  * Replay Buffer
  * Target Network
  * ε-greedy exploration

---

## 🏗️ Project Structure

```text
proj_1/
│
├── flappy_bird.py      # Environment (game logic + reward)
├── dqn.py              # Neural network
├── agent.py            # DQN agent
├── replay_buffer.py    # Experience replay
├── train.py            # Training loop (2000 episodes)
├── grid_search.py      # Hyperparameter tuning
├── main.py             # Manual / demo play
└── assets/             # Images (bird, etc.)
```

---

## ▶️ Πώς να το τρέξεις

### 1. Εγκατάσταση dependencies

```bash
pip install torch numpy pygame matplotlib
```

---

### 2. Training (εκπαίδευση agent)

Από τον φάκελο `proj_1`:

```bash
python train.py
```

👉 Αυτό θα:

* τρέξει ~2000 episodes
* αποθηκεύσει:

  * `best_flappy_model.pth`
  * `final_flappy_model.pth`
* δημιουργήσει plots:

  * `training_scores.png`
  * `training_rewards.png`

---

### 3. Manual Play (δοκιμή περιβάλλοντος)

```bash
python main.py
```

Controls:

* `SPACE`: flap

---

## ⚙️ Training Configuration

Best configuration (από grid search):

```python
lr = 5e-4
epsilon_decay = 0.998
gamma = 0.95
distance_weight = 0.01
episodes = 2000
```

---

## 🔍 Hyperparameter Tuning

```python
param_grid = {
    "lr": [1e-3, 5e-4],
    "epsilon_decay": [0.995, 0.998],
    "gamma": [0.99, 0.95],
    "distance_weight": [0.01, 0.02]
}
```

---

## 🧪 Reward Design

* +1 survival
* +10 pipe
* -100 death
* penalty απόστασης από gap
* μικρό penalty για flap

---

## 📈 Αποτελέσματα

* Σταδιακή βελτίωση performance
* Αύξηση average score μετά από tuning
* Visualization μέσω learning curves

---

## ⚠️ Limitations

* Χαμηλό τελικό score
* Ευαισθησία σε reward shaping
* Περιορισμένος χρόνος εκπαίδευσης

---

## 🚀 Future Work

* Double DQN
* Dueling DQN
* CNN (image-based input)
* Improved reward design

---

## 📚 Technologies

* Python
* PyTorch
* NumPy
* Pygame

---

## 👤 Author

Andreas Darsaklis
Reinforcement Learning Project – AUTH
