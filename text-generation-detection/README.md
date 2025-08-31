# Text Generation Detection System

**Course:** COMP90051 Statistical Machine Learning  
**Project Type:** Group Project (Kaggle Competition)  
**Topic:** Detecting whether text is **human-written** or **machine-generated** across multiple domains.  

---

## Project Goal
With the rise of large language models (LLMs), distinguishing human-written text from AI-generated text has become an important challenge.  
This project aimed to build robust classifiers to **detect AI-generated text**, even under **domain shift** and **imbalanced label distributions**.  

---

## Dataset
- **Domain 1:** Balanced dataset (500 human, 500 machine).  
- **Domain 2:** Imbalanced dataset (250 human, 4750 machine).  
- **Test set:** 4000 samples, balanced across domains and classes.  
- Task: Predict binary label → `0 = human` / `1 = machine`.  

---

## Approach
We explored multiple models and integrated them into an **ensemble system**:  
1. **TF-IDF + MLP** → Strong baseline for sparse text representation.  
2. **CNN** → Captures local n-gram features.  
3. **BiLSTM** → Captures long-range dependencies and sequence structure.  
4. **Ensemble** → Combined predictions from CNN, TF-IDF MLP, and BiLSTM for improved robustness.  

### Key Challenges Tackled
- **Domain adaptation** → Trained models to generalize across unseen domains.  
- **Imbalanced classification** → Used sampling techniques and loss adjustments to balance performance on minority class (human-written samples).  
- **Error analysis & ablation studies** → Evaluated the contribution of each model and preprocessing step.  

---

## Results
- **Baseline (Bag-of-words + Logistic Regression):** ~65% accuracy.  
- **Our Ensemble (CNN + TF-IDF MLP + BiLSTM):** **92.63% accuracy** on test set.  
- Achieved **Top-10 leaderboard ranking** in Kaggle competition.  

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** PyTorch, Scikit-learn, NumPy, Pandas  
- **Techniques:** Ensemble Learning, CNN, BiLSTM, TF-IDF + MLP, Domain Adaptation, Imbalanced Classification  
- **Tools:** Kaggle, Google Colab, Git, Jupyter Notebook  

---

## Future Work
- Explore transformer-based models (e.g., DistilBERT) for stronger representation.  
- Apply adversarial domain adaptation techniques to further improve generalization.  
- Extend to multi-class detection (distinguishing *which* LLM generated the text).  