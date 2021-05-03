# Projekt 6 - Deep Learning

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

**Die Daten werden nun in Trainings- und Test gesplittet. Dazu sollte zunächst definiert werden was das X-Array (Daten mit den Features) und was das y-Array (Daten mit der Zielvariable) ist:** 
X=df.drop("loan_repaid",axis=1).values
y=df["loan_repaid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
**Durch Ausführen der obigen Funktionen werden die Daten normalisiert und transformiert

**Die Leistung des Modells kann durch ein Plot dargestellt werden, in dem der loss der Validierung und der loss des Trainings verglichen werden:**
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
**Im Diagramm sollte zu sehen sein, dass die orangene Linie (entspricht den echten Daten) fast einer geraden Linie entspricht. Die blaue Linie (tatsächlicher loss des Modells) hingegen geht im ersten Bereich stark runter.
