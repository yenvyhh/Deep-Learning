# Projekt 6 - Deep Learning
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yenvyhh/K-Means-Clustering/main?filepath=K%20Means%20Clustering%20-%20Projekt%203.ipynb)

**Die Daten importieren und als Data Frame abspeichern:**
df = pd.read_csv('../DATA/lending_club_loan_two.csv')
       
**Informationen des Data Frames bzw. der Daten anzeigen lassen:**     
df.info()
**Bei Info wird angezeigt, ob die Spalten einen Float, ein Integer oder ein Object sind. Zu dem wird bei RangeIndex angezeigt, dass es 396030 Einträge gibt.

**Darauffolgend erfolgt eine EXPLORATIVE DATENANALYSE, die durch verschiedene Diagrammvisualisierungen dargestellt werden. Ein Beispiel, das ausgeführt wird:**
sns.countplot(data=df, x="loan_status")
**Durch Ausführen wie ein Countplot erstellt. Es ist die Verteilung der Anzahl basierend auf dem loan_status zu erkennen. Dieser ist aufgeteilt in "Fully Paid" und "Charged Off".

**Ein weiteres Beispiel soll die Korrelation zwischen den Featurevariablen anzeigen:**
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)
plt.ylim(10, 0)
**Die Heatmap zeigt die verschiedenen Korrelationsgrößen zwischen den Featurevariablen an. Im Beispiel ist zu erkennen, dass zwischen "installment" und "loan_amt" eine hohe Korrelation vorhanden ist (Wert= 0.95).

**Im nächsten Schritt wird die Datenaufbereitung durchgeführt.**
df.isnull().sum()
**Die Serie zeigt an, wie viele Werte in den jeweiligen Spalten fehlen. Im vorliegenden Projekt fehlen z. B. in den Spalten "emp_title (22927 fehlende Werte)","emp_length (18301 fehlende Werte)" und "title (1755 fehlende Werte)" Werte. Die fehlenden Werte werden im Verlauf des Projektes entfernt oder ersetzt.
Ein Beispiel zur Entfernung von Zeilen, die keine Werte beinhalten, wird wie folgt durchgeführt:**
df= df.dropna()

**Im Anschluss werden die kategorischen daten umgewandelt.**
df.select_dtypes(['object']).columns
**Diese Funktion zeigt uns, welche Spalten/Features ein Object sind und umgewandelt werden müssen.**

subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df= pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.head()
**Durch Ausführen der drei obigen funktionen, werden die Object in "sub_grade" in dummies umgewandelt, die nur die Werte 0 oder 1 annehmen können. Daraufhin werden das bestehende Data Frame und das neue "subgrade_dummies" zusammengefügt und die alte Spalte "sub_grade" entfernt. Im head sollten die Spalten nun von A2 bis G5 zu sehen sein.



**Basierend darauf kann ein Klassifizierungsreport und eine Confusion Matrix für das Modell erstellt werden:**
print (confusion_matrix(df["Cluster"],kmeans.labels_))
print ("\n")
print (classification_report(df["Cluster"],kmeans.labels_))
**Je näher die Werte bei precicion, recall und f1-score an 1 sind, desto genauer sind Auswertung. **
