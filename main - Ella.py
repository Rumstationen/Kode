

#Go here to the URL shown beneath and navigate to the online  test tool under the headline section:
# 'Accessing the Astro Pi Replay Tool online ' the online tool is called ' Astro Pi Replay Tool' 

# https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2
# Import the Camera class from the picamera-zero module
from exif import Image#Den hente programerings kode den skal bruge længere ned
from datetime import datetime#Henter kode
import cv2#Importerer cv2 pakke
import math#Importerer math pakke

from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# Capture an image
"#Tager flere billeder for at det tager længere tid"#Flere billeder skaber længere tid mellem billeder
cam.take_photo("image1.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")


cam.take_photo("image2.jpg")


image_1 = 'image1.jpg'
image_2 = 'image2.jpg'

def get_time(image):#Finder tidspunktet billedet er taget
    with open(image, 'rb') as image_file:#Åbner billedet og konverterer til image objekt
        img = Image(image_file)#Tager filen og lægger den i img
        for data in img.list_all():
            print(data)
            time_str = img.get("datetime_original")#Gemmes som en streng
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')#Konverterer til datetime objekt
    return time#Returnere variablen vi har lavet som time.
#print(get_time('photo_0683.jpg'))

def get_time_difference(image_1, image_2):#Skal finde tidsforskellen mellem billederne
    #De næste to linjer finder tiden for de to billeder
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1#Finder tidsforskellen mellem billederne ved at trække tiderne fra hinanden
    return time_difference.seconds#Retunerer time i sekunder
#    print(time_difference)
#Konverterer billederne til opencv
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv
#Finder kendetegn i de to billeder
def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2
#Finder ens kendetegn ved at bruge brute force
"""Brute force er når det går igennem alle kombinationer. Det vil sige at den går gennem alle kendetegnene i hvert billede
og ser hvilke nogle der er ens"""
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
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



image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects

time_difference = get_time_difference(image_1, image_2) # Get time difference between images
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
#display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches

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