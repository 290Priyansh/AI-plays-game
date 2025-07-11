# 🎮 Doodle Jump AI

A fully playable **Doodle Jump-style game** built using `pygame`, enhanced with a **Q-learning-based AI agent**. The AI learns to play the game through reinforcement learning and can switch between AI and manual modes during gameplay.

---

## 🧠 Features

- 🎮 **Manual Mode**: Control the doodle with keyboard arrows or `A/D`.
- 🤖 **AI Mode**: Enable AI to play using a trained **Q-table**.
- 🦘 Platform-to-platform jumping logic with sticky targeting.
- 🌀 Spring platforms give extra jump boost.
- 🧠 AI shows its decisions via a red line to the next target platform.
- 📊 Collect training data for future model improvements.

---

## 🚀 How It Works

The agent uses **Q-learning** and/or a supervised model to decide between:
- `LEFT`
- `RIGHT`
- `NONE`

The **state** consists of:
- Horizontal and vertical distance to next platform
- Velocity
- Spring presence
- Directional hint
- Stuck loop detection

The AI picks actions based on a trained Q-table (`q_table.pkl`), and can be enhanced with supervised models like `supervised_model.pkl`.

---

## 🎥 Demo

![Model_train Demo](demo/model_train.gif)
![Gameplay_Demo](demo/model_train.gif)


