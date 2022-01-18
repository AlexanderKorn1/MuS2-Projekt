# MuS2-Projekt
Unsupervised Domain Adaptation


Ausführen der implementierten Lösung:

Herunterladen der Dateien aus dem Git-Respository: https://github.com/AlexanderKorn1/MuS2-Projekt

Entpacken des "pics" Zip Folders (dieser beinhaltet das Trainingsset mit realen Küchenbildern)

Terminal starten und in das Verzeichnis des Projektordners wechseln


Ausführen der gewünschten WGAN Implementierung mit dem Befehl:

python3 wgan_mus.py oder python3 wgan_gp_mus.py

Die Hyperparameter des WGANs können entweder direkt im Programmcode oder im Terminal beim Programmaufruf via Argumente verändert werden

Die generierten/synthetisierten Bilder werden in einem eigens erstellen "images" Folder abgespeichert.

Sollen andere Bilder synthetisiert werden, müssen in dem "pics" Folder die entsprechenden Domänenbilder ersetzt werden.