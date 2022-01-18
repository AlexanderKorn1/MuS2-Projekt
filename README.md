# MuS2-Projekt
Unsupervised Domain Adaptation

durchgeführt von: Haas, Korn, Scherl

Quelle für PyTorch-GANs: https://github.com/eriklindernoren/PyTorch-GAN

In diesem Projekt werden Bilder einer Zieldomäne mithilfe von Generative Adversarial Networks (GANs) synthetisiert.
Hierfür werden verschiedene Wasserstein-GAN (WGAN) implementiert, um synthetische Bilder zu generieren.
Als Datensatz wird der "Indoor Scene Recognition" Datensatz (Quelle: http://web.mit.edu/torralba/www/indoor.html) verwendet, wobei die Kategorie "kitchen" für dieses Projekt gewählt wird.

# Vorbereitung

Herunterladen der Dateien aus dem Git-Repository: https://github.com/AlexanderKorn1/MuS2-Projekt

Entpacken des "pics" Zip Folders (dieser beinhaltet das Trainingsset mit realen Küchenbildern)

Terminal starten und in das Verzeichnis des Projektordners wechseln


# Ausführen der gewünschten WGAN Implementierung

Befehl: python3 wgan_mus.py oder python3 wgan_gp_mus.py

Die Hyperparameter des WGANs können entweder direkt im Programmcode oder im Terminal beim Programmaufruf via Argumente verändert werden

Die generierten/synthetisierten Bilder werden in einem eigens erstellen "images" Folder abgespeichert.

Sollen andere Bilder synthetisiert werden, müssen in dem "pics" Folder die entsprechenden Domänenbilder ersetzt werden.