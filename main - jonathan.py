#Go here to the URL shown beneath and navigate to the online  test tool under the headline section:
# 'Accessing the Astro Pi Replay Tool online ' the online tool is called ' Astro Pi Replay Tool' 

# https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2
# Import the Camera class from the picamera-zero module
from exif import Image# henter kode for Image
from datetime import datetime# henter kode fra bibilotek
import cv2# henter cv2 pakken
import math# henter math pakken

from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# Capture an image

cam.take_photo("image1.jpg")#importer billede image1.jpg
cam.take_photo("image2.jpg")#importer billede image2.jpg
cam.take_photo("image2.jpg")#risser på pladen
cam.take_photo("image2.jpg")#risser på pladen
cam.take_photo("image2.jpg")#risser på pladen


cam.take_photo("image2.jpg")#risser på pladen


image_1 = 'image1.jpg'#bestemmer image1.jpg som image_1
image_2 = 'image2.jpg'#bestemmer image2.jpg som image_2


def get_time(image):#definere get_time for image vilkårlig
    with open(image, 'rb') as image_file:# åbner billede filen 
        img = Image(image_file)#propper Image(image_file) ind i img
        for data in img.list_all():# for data i img filer list alle
            print(data)# vis mig dataen
            time_str = img.get("datetime_original")#danner en tråd med dataen for da billede blev taget
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')#fjerner det data som ikke er vigtigt for time_str 
    return time# slutter funktionen og sætter den definererede tid ind i programmet
#print(get_time('photo_0683.jpg'))

def get_time_difference(image_1, image_2): #definerer get_time_difference(image_1, image_2)
    time_1 = get_time(image_1)#propper den hentede tid fra image_1 ind i time_1 
    time_2 = get_time(image_2)#propper den hentede tid fra image_2 ind i time_2
    time_difference = time_2 - time_1# trækker den time_1 fra time_2 og sætter resultatet som time_difference
    return time_difference.seconds# slutter funktionen og sætter tidsdefferencen ind i programmet målt i sekunder
#    print(time_difference)
def convert_to_cv(image_1, image_2):# definere convert_to_cv(image_1, image_2)
    image_1_cv = cv2.imread(image_1, 0)# convertere image_1 til cv2, og propper den ind i image_1_cv2
    image_2_cv = cv2.imread(image_2, 0)# convertere image_2 til cv2, og propper den ind i image_2_cv2
    return image_1_cv, image_2_cv# slutter funktionen og sætter image_1_cv, image_2_cv ind i programmet
def calculate_features(image_1, image_2, feature_number):# definere calculate_features(image_1, image_2, feature_number)
    orb = cv2.ORB_create(nfeatures = feature_number)# sætter feature-punkter på billede
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)# sætter keypoints_1 og descriptors_1 for image_1_cv
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)# sætter keypoints_2 og descriptors_2 for image_1_cv
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2# slutter funktionen og sætter keypoints_1, keypoints_2, descriptors_1, descriptors_2 ind i programmet
def calculate_matches(descriptors_1, descriptors_2):# definere calculate_matches(descriptor_1, descriptor_2)
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)# tjek hele lortet en pixel af gangen 
    matches = brute_force.match(descriptors_1, descriptors_2)# sammenlign descriptors_1 og desciptors_2
    matches = sorted(matches, key=lambda x: x.distance)# 
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
