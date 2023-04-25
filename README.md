# Realiser-une-application-d-estimation-de-prix-d-une-voiture
Votre client, un revendeur de voiture, souhaite la création d'une application pouvant estimer le prix d'une voiture.

# Définitions
car_ID : un identifiant unique pour chaque voiture dans l'ensemble de données.
symboling : le niveau de risque d'assurance de la voiture, où -2 est le plus risqué et +3 est le moins risqué.
CarName : le nom de la voiture, comprenant à la fois la marque et le modèle.
fueltype : le type de carburant utilisé par la voiture (soit "essence" ou "diesel").
aspiration : si la voiture est à aspiration naturelle ou turbocompressée.
doornumber : le nombre de portes de la voiture (soit "deux" ou "quatre").
carbody : le style de carrosserie de la voiture (par exemple, berline, hatchback, convertible, etc.).
drivewheel : le type de transmission utilisé par la voiture (par exemple, transmission à traction avant, transmission à propulsion arrière, transmission intégrale).
enginelocation : l'emplacement du moteur (soit "avant" ou "arrière").
wheelbase : la distance entre les roues avant et arrière de la voiture.
carlength : la longueur totale de la voiture.
carwidth : la largeur totale de la voiture.
carheight : la hauteur totale de la voiture.
curbweight : le poids de la voiture sans aucun occupant ni cargaison.
enginetype : le type de moteur utilisé par la voiture (par exemple, quatre cylindres, six cylindres, rotatif, etc.).
cylindernumber : le nombre de cylindres dans le moteur de la voiture.
enginesize : la taille du moteur de la voiture en centimètres cubes (cc).
fuelsystem : le type de système de carburant utilisé par la voiture (par exemple, carburé, à injection de carburant).
boreratio : le rapport du diamètre des cylindres du moteur à leur longueur.
stroke : la distance parcourue par le piston en montée et en descente dans les cylindres du moteur.
compressionratio : le rapport du volume de la chambre de combustion du moteur lorsque le piston est en bas de sa course par rapport à lorsqu'il est en haut de sa course.
horsepower : la puissance du moteur de la voiture en chevaux-vapeur (cv).
peakrpm : la vitesse de rotation du moteur à laquelle la puissance maximale de la voiture est produite.
citympg : l'économie de carburant de la voiture en miles par gallon (mpg) en conditions de conduite en ville.
highwaympg : l'économie de carburant de la voiture en miles par gallon (mpg) en conditions de conduite sur autoroute.
price : le prix de détail suggéré par le fabricant (MSRP) de la voiture en dollars américains.

# Voici les conversions pour transformer les unités américaines en unités françaises :

    empattement : en mètre (1 pouce = 0,0254 mètre)
    longueur : en mètre
    largeur : en mètre
    hauteur : en mètre
    poids_vehicule : en kilogramme (1 livre = 0,453592 kilogramme)
    taille_moteur : en litre (1 pouce cube = 0,0163871 litre)
    taux_alésage : en millimètre (1 pouce = 25,4 millimètres)
    course : en millimètre
    taux_compression : en ratio (pas de conversion nécessaire)
    tour_moteur : en tours par minute (pas de conversion nécessaire)
    consommation_ville : en litres aux 100 km (1 mile par gallon = 0,425 km par litre)
    consommation_autoroute : en litres aux 100 km
