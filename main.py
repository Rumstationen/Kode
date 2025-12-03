#Go here to the URL shown beneath and navigate to the online  test tool under the headline section:
# 'Accessing the Astro Pi Replay Tool online ' the online tool is called ' Astro Pi Replay Tool' 
 
# https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2
# Import the Camera class from the picamera-zero module
"""
exif står for "excangeable image file format". Exif data viser en masse informationer om billedet såsom
hvornår det blev taget, lokationen og kameratypen. man kan fremskaffe disse informationer ved
at skrive exif import image. på denne måde tager man informationerne fra billedet.
Derefter skal man skrive "datetime import datetime". Det er vigtigt at, de billeder man bruger ligger i
samme folder som den kode, som man er i gang med.

For at finde ud af hvornår billedet blev taget skal man
skrive "def get_time(image)". For at lave billedet om til et billede objekt skal man skrive
"with open(image,'rb') as image_file"
"""
from exif import Image
#på denne måde tager man informationerne fra billedet. eks. hvor og hvornår billedet er blevet taget eller med hvilket kamera
# her defineres det som en funktion, som kan bruge exif
from datetime import datetime
#henter billedet ned
import cv2
#importere cv2-pakken
import math
#importere Matematik-pakken
from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# tag billede

cam.take_photo("image1.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")

#tager billederne
cam.take_photo("image2.jpg")


image_1 = 'image1.jpg'
image_2 = 'image2.jpg'
#definerer navnene

def get_time(image):
    with open(image, 'rb') as image_file:
        #at åbne den for at læse filen
        img = Image(image_file)
        #definere det, så vi "bare" kan skrive img i stedet
        for data in img.list_all():
            print(data)
            time_str = img.get("datetime_original")
#str er en bogstavvariabel for streng
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time
    #Returnere variablen der er lavet som time.

#print(get_time('photo_0683.jpg'))

def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    #definerer ny funktion og henter tiderne fra Exif-data til begge billeder
    time_difference = time_2 - time_1
    #trækker tiderne fra hinanden for at finde forskellen
    return time_difference.seconds
# for at få tiden i sekunder
#print(time_difference)

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    #definere de to billeder som OpenCV-objekter
    return image_1_cv, image_2_cv
#retunerer objekterne. 
def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2
# de ovenstående linjer er til at finde nøglepunkter og diskriptioner for billederne
#til dette bruges ORB-algoritmen
"""
Brute Force betyder, at man tager alle diskriptioner fra billeder for at finde alle matches i et andet billede.
"""
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
# De ovenstående koder forsøger at finde matches v.h.a. rute Force

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    #tegn linjer imellem nøglepunkterne, når de er ens på de to billeder.
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    #ændre størrelse på billederne og vise dem ved siden af hinanden med de indtegnede streger
    cv2.waitKey(0)
    # vent til der bliver trykket på tasten 
    cv2.destroyWindow('matches')
    # afslut funktionen når tasten trykkes på

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    # her oprettes der en ny funktion, som kan tage de to sæt nøglepunkter og listen. disse bruges som argumenter
    coordinates_1 = []
    coordinates_2 = []
    #oprettelse af to tomme lister, så koordinaterne kan gemmes.
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        # for-løkken hjælper med at hente x1,y1,x2 og y2 koordinaterne for begge billederne
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)
    #print(coordinates_1[0])
    #print(coordinates_2[0])
    #print(merged_coordinates[0])
def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed



image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # laver OpenCV billede-objekter

time_difference = get_time_difference(image_1, image_2) # finder tidsforskel mellem billederne
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # laver OpenCV billede-objekter
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # få keypoints og descriptors
matches = calculate_matches(descriptors_1, descriptors_2) # sammenlign descriptors
#display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # vis steder hvor det er det samme

coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)


#timedif=get_time_difference('image1.jpg', 'image2.jpg')


# Format the estimate_kmps to have a precision
# of 5 significant figures
#timedifFormatted = "{:.3f}".format(timedif)

#file_path = "result.txt"  # Replace with your desired file path
#with open(file_path, 'w') as file:
#    file.write(timedifFormatted)

estimate_kmps = speed  # Replace with your estimate

# Format the estimate_kmps to have a precision
# of 5 significant figures
estimate_kmps_formatted = "{:.3f}".format(estimate_kmps)

# Create a string to write to the file
output_string = estimate_kmps_formatted
#print(output_string)
# Write to the file
file_path = "result.txt"  # Replace with your desired file path
with open(file_path, 'w') as file:
    file.write(output_string)

#print("Data written to", file_path)
