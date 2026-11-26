# Regression und Maschinelles Lernen

Hier wird einfach erklärt, warum die Regression des QIT (Intelligenzquotient) in diesem Projekt zum **Maschinellen Lernen** gehört und nicht nur zu einer einfachen statistischen Auswertung.

Die Grundlage dafür ist der Praktikumsbericht *Rapport SFE Samuel Fornet* (Année scolaire 2026/2027). Alle Textstellen beziehen sich auf die Abschnitte (§) dieses Berichts.

---

## 1. Was war das Ziel?

Das Ziel des Projekts war es, aus frühen HRV-Daten (Herzfrequenzvariabilität aus dem EKG von Neugeborenen) den **QIT im Vorschulalter vorherzusagen**. Der QIT wird mit dem WPPSI-III-Test gemessen.

Im Bericht heißt es dazu (Sinngemäß, siehe §1 und §4):
> „L’objectif de ce stage était de développer et d’évaluer des modèles d’apprentissage automatique capables d’estimer, à partir d’indices de HRV enregistrés précocement, le quotient intellectuel total (QIT) mesuré vers l’âge de 3 ans.“ ([Rapport SFE Samuel Fornet, §1])

> „L’objectif principal était d’évaluer si les indices de variabilité de la fréquence cardiaque […] permettaient de prédire le quotient intellectuel total obtenu à la WPPSI-III à l’âge de 3 ans.“ ([Rapport SFE Samuel Fornet, §4])

Das ist eine klassische Aufgabe des **überwachten maschinellen Lernens**: Ein Modell lernt aus Trainingsdaten, wie Eingabemerkmale (hier HRV-Indizes) mit einer Zielgröße (hier QIT) zusammenhängen, und wendet das Gelernte dann auf neue, noch nicht gesehene Daten an ([Rapport SFE Samuel Fornet, §1, §4]).

---

## 2. Warum Regression und keine Klassifikation?

Der QIT ist eine **kontinuierliche Zahl** (durchschnittlich 100, Standardabweichung 15). Man könnte ihn auch in Kategorien aufteilen (z. B. „unter 90“ und „über 90“), aber der Bericht sagt dazu (§5.4.5):

> „La prédiction du QIT continu par régression constituait l’analyse méthodologiquement prioritaire. Elle évitait la perte d’information induite par la dichotomisation…“ ([Rapport SFE Samuel Fornet, §5.4.5])

Außerdem wird im §3.2 erwähnt, dass der QIT standardisiert ist (Mittelwert 100, Standardabweichung 15) und als kontinuierliche Variable betrachtet werden sollte. Eine Aufteilung in nur zwei Gruppen würde wertvolle Informationen verlieren ([Rapport SFE Samuel Fornet, §3.2, §5.4.5]).

Daher wurde primär **regressioniert**, also eine Zahl vorhergesagt, statt nur eine Schublade zuzuordnen.

---

## 3. Welche Modelle wurden benutzt?

Im Bericht werden verschiedene Machine-Learning-Modelle für die Regression genannt (§5.3.4), zum Beispiel:

- Lineare Regression
- Ridge und Lasso (Regression mit Straftermen gegen Überanpassung)
- Random Forest (Viele Entscheidungsbäume zusammen)
- SVR (Support Vector Regression)
- MLP (Künstliches neuronales Netz)

Besonders wichtig: Modelle wie Lasso oder Ridge wurden hier nicht einfach „so angewendet“, sondern mit automatischer **Hyperparameter-Optimierung** (Grid Search) und **Cross-Validation** ausgestattet ([Rapport SFE Samuel Fornet, §5.3.3, §5.3.4]). Das macht sie zu Machine-Learning-Modellen und nicht zu einfachen statistischen Rechnungen.

Auch im §3.3.4 wird ausdrücklich erwähnt, dass HRV-Indizes kompatibel sind mit „méthodes supervisées telles que la régression pénalisée, les machines à vecteurs de support, les forêts aléatoires ou le gradient boosting“ ([Rapport SFE Samuel Fornet, §3.3.4]).

---

## 4. Warum ist das Machine Learning und nicht nur Schul-Statistik?

Ein einfaches lineares Rechenmodell auf einem Datensatz ohne Aufteilung in Training und Testdaten ist eher klassische Statistik.

Hier wurde aber ein vollständiges **ML-Pipeline** verwendet, wie im Bericht beschrieben:

1. Daten wurden aufgeteilt in Training und Test (einfache Trennung, dann 5-fache Cross-Validation, dann verschachtelte Cross-Validation) ([Rapport SFE Samuel Fornet, §5.3.1, §5.3.2, §5.3.3]).
2. Hyperparameter wurden automatisch auf internen Falten optimiert (Grid Search in der inneren Schleife der verschachtelten Cross-Validation) ([Rapport SFE Samuel Fornet, §5.3.3]).
3. Die Leistung wurde auf **unabhängigen Testdaten** gemessen (Metriken: MAE, RMSE, R²) ([Rapport SFE Samuel Fornet, §5.4.1]).

Genau das unterscheidet Machine Learning von einfacher deskriptiver Statistik.

---

## 5. Fazit für den Lehrer

Die Aussage „Regression gehöre nicht zum Machine Learning“ trifft auf diese Arbeit **nicht zu**.

Der Bericht zeigt eindeutig:
- Es wurde vorhergesagt, nicht nur beschrieben ([Rapport SFE Samuel Fornet, §1, §4, §5.4.5]).
- Es wurden ML-Modelle (Lasso, Ridge, Random Forest, SVR etc.) verwendet ([Rapport SFE Samuel Fornet, §3.3.4, §5.3.4]).
- Die Modelle wurden mit Cross-Validation validiert und auf Testdaten geprüft ([Rapport SFE Samuel Fornet, §5.3.3, §5.4.1]).

Regression ist in diesem Fall also ein vollwertiger Teil des maschinellen Lernens – eben **prädiktive Regression mit Machine-Learning-Methoden**.
