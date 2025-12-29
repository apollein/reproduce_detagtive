import json
from itertools import islice
import statistics
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
from scipy import ndimage
from math import ceil
import argparse

global TIME_MIN_THRESHOLD
global TIME_MAX_THRESHOLD
global TIME_MIN_GLOBAL
global TIME_MAX_GLOBAL
global MEAN_ADV_PERIOD_THRESHOLD
global MEAN_RSSI_THRESHOLD
global WINDOW_LENGTH
global POLYORDER
global SAVGOL_ENABLED
global GAUSSIAN_ENABLED
global GAUSSIAN_SIGMA

LEN_THRESHOLD = 100
TIME_MIN_THRESHOLD = 60*5
TIME_MAX_THRESHOLD = 60*40
TIME_MIN_GLOBAL = 42
TIME_MAX_GLOBAL = 42
MEAN_ADV_PERIOD_THRESHOLD = 1
MEAN_RSSI_THRESHOLD = -75

SAVGOL_ENABLED = False
WINDOW_LENGTH = 20        # window length
POLYORDER = 3  # cubic polynomial approximation

GAUSSIAN_ENABLED = True
GAUSSIAN_SIGMA = 5

measurements = {}
raw_measurements = {}
detected_devices = {}

def set_global_time_intervals(verbose=True):
    TIME_MIN_GLOBAL = None
    TIME_MAX_GLOBAL = None
    for device in measurements:
        if TIME_MIN_GLOBAL==None:
            TIME_MIN_GLOBAL = measurements[device]["start"]
        else:
            if measurements[device]["start"]<TIME_MIN_GLOBAL:
                TIME_MIN_GLOBAL = measurements[device]["start"]
        if TIME_MAX_GLOBAL==None:
            TIME_MAX_GLOBAL = measurements[device]["end"]
        else:
            if measurements[device]["end"]>TIME_MAX_GLOBAL:
                TIME_MAX_GLOBAL = measurements[device]["end"]
    if verbose:
        print("First received signal: {}".format(TIME_MIN_GLOBAL))
        print("Last received signal: {}".format(TIME_MAX_GLOBAL))
        print("")
    return (TIME_MIN_GLOBAL, TIME_MAX_GLOBAL)

def filter_device(device_measurements):
    if len(device_measurements)<LEN_THRESHOLD:
        return False
    if (device_measurements[-1][0] - device_measurements[0][0])<TIME_MIN_THRESHOLD:
        return False
    if (device_measurements[-1][0] - device_measurements[0][0])>TIME_MAX_THRESHOLD:
        return False
    if float(np.mean(device_measurements, axis=0)[1])<MEAN_RSSI_THRESHOLD:
        return False
    if float(np.mean(np.diff(np.take(device_measurements, 0, axis=1))))>MEAN_ADV_PERIOD_THRESHOLD:
        return False
    return True

def matchScore(mean_rssi_cur, stdev_rssi_cur, mean_time_cur, mean_rssi_can, stdev_rssi_can, mean_time_can):
    return abs(mean_rssi_cur - mean_rssi_can) + abs(stdev_rssi_cur - stdev_rssi_can) + abs(mean_rssi_cur - mean_rssi_can)

def search_successor(current_mac):
    search_interval = measurements[current_mac]["mean_adv_period"] * 10
    s1 = None
    score = 42
    for sample in measurements:
        if (measurements[sample]["start"] > measurements[current_mac]["end"]) and (measurements[sample]["start"] < (measurements[current_mac]["end"]+timedelta(seconds=search_interval))):
            if s1 == None:
                s1 = sample
                score = matchScore(measurements[current_mac]["mean_rssi"], measurements[current_mac]["stdev_rssi"], measurements[current_mac]["mean_adv_period"], measurements[sample]["mean_rssi"], measurements[sample]["stdev_rssi"], measurements[sample]["mean_adv_period"])
            else:
                if matchScore(measurements[current_mac]["mean_rssi"], measurements[current_mac]["stdev_rssi"], measurements[current_mac]["mean_adv_period"], measurements[sample]["mean_rssi"], measurements[sample]["stdev_rssi"], measurements[sample]["mean_adv_period"])<score:
                    s1 = sample
                    score = matchScore(measurements[current_mac]["mean_rssi"], measurements[current_mac]["stdev_rssi"], measurements[current_mac]["mean_adv_period"], measurements[sample]["mean_rssi"], measurements[sample]["stdev_rssi"], measurements[sample]["mean_adv_period"])
    return s1

def print_devices(verbose=True):
    for device in detected_devices:
        print("Device {}: {} addr".format(device, len(detected_devices[device])))
        for mac_addr in detected_devices[device]:
            print(" - {} from {} to {}".format(mac_addr, measurements[mac_addr]["start"], measurements[mac_addr]["end"]))
            if verbose:
                print("    -> Mean RSSI of {} dB".format(measurements[mac_addr]["mean_rssi"]))
                print("    -> Std dev RSSI of {} dB".format(measurements[mac_addr]["stdev_rssi"]))
                print("    -> Mean Adv period of {} s".format(measurements[mac_addr]["mean_adv_period"]))
        print("")



def draw_device(device_id, fig, ax, stdcolor="blue", savgol=False, gaussian=True):
    for mac_addr in detected_devices[device_id]:
        x_value = np.take(measurements[mac_addr]["dt_measurements"], 0, axis=1)
        if savgol:
            y_value = savgol_filter(np.take(measurements[mac_addr]["measurements"], 1, axis=1), int(np.max([WINDOW_LENGTH, np.min([50, np.rint(np.divide(20,measurements[mac_addr]["mean_adv_period"]))])])), POLYORDER, mode='nearest')
        elif gaussian:
            y_value = ndimage.gaussian_filter1d(np.take(measurements[mac_addr]["measurements"], 1, axis=1), GAUSSIAN_SIGMA)
        else:
            y_value = np.take(measurements[mac_addr]["measurements"], 1, axis=1)
        ax.plot(x_value, y_value, linewidth=1, label="{} - {}".format(device_id, mac_addr))
        ax.fill_between(x=x_value, y1=y_value-measurements[mac_addr]["stdev_rssi"], y2=y_value+measurements[mac_addr]["stdev_rssi"], color=stdcolor,  interpolate=True, alpha=.05)

def draw_devices(devices_id, savgol=False, gaussian=True):
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan', 'pink', 'gray', 'olive']
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    format_str = '%H:%M:%S'
    format_ = mdates.DateFormatter(format_str)
    ax.xaxis.set_major_formatter(format_)
    for device in range(len(devices_id)):
        draw_device(devices_id[device], fig, ax, stdcolor=colors[device%len(colors)], savgol=SAVGOL_ENABLED, gaussian=GAUSSIAN_ENABLED)
    ax.legend()
    plt.show()

def select_devices():
    to_draw = []
    for device in detected_devices:
        if len(detected_devices[device])>2:
            to_draw.append(device)
    return to_draw

def get_data_by_device(input):
    output_simple = {}
    output_raw = {}
    for element in input:
        if 'platform_data' in element:
            if 'Address' in element['platform_data']:
                if element['platform_data']['Address'] not in output_simple:
                    output_simple[element['platform_data']['Address']] = []
                    output_raw[element['platform_data']['Address']] = []
                output_simple[element['platform_data']['Address']].append([element['timestamp'], element['rssi']])
                output_raw[element['platform_data']['Address']].append(element)
    return output_simple, output_raw

def get_data(file):
    with open(file, 'r') as f:
        input = json.load(f)
        input_measurements, raw_measurements = get_data_by_device(input)
        for device in input_measurements:
            if filter_device(input_measurements[device]):
                measurements[device] = {"measurements": [], "dt_measurements": [], "mean_rssi": 0, "stdev_rssi": 0, "mean_adv_period": 0, "emission_length": 0}
                for measurement in input_measurements[device]:
                    measurements[device]["measurements"].append((measurement[0], measurement[1]))
                    measurements[device]["dt_measurements"].append((datetime.fromtimestamp(measurement[0]), measurement[1]))
                measurements[device]["mean_rssi"] = float(np.mean(measurements[device]["measurements"], axis=0)[1])
                measurements[device]["stdev_rssi"] = float(np.std(measurements[device]["measurements"], axis=0)[1])
                measurements[device]["mean_adv_period"] = float(np.mean(np.diff(np.take(measurements[device]["measurements"], 0, axis=1))))
                measurements[device]["start"] = datetime.fromtimestamp(measurements[device]["measurements"][0][0])
                measurements[device]["end"] = datetime.fromtimestamp(measurements[device]["measurements"][-1][0])
                measurements[device]["emission_length"] = timedelta(seconds=(measurements[device]["measurements"][-1][0] - measurements[device]["measurements"][0][0]))
        return input_measurements, raw_measurements

def compare_metadata_between_ids(device):
    if device in detected_devices:
        for d in detected_devices[device]:
            print("device id: {}, packet {}".format(d, 0))
            print("manufacturer data: {}".format(raw_measurements[d][0]['manufacturer_data']))
            print(raw_measurements[d][0]['platform_data'])
            end = len(raw_measurements[d])-1
            print('')
            print("device id: {}, packet {}".format(d, end))
            print("manufacturer data: {}".format(raw_measurements[d][end]['manufacturer_data']))
            print(raw_measurements[d][end]['platform_data'])
            print("------------------------")

def successors_quest():
    for device in measurements:
        search_result = search_successor(device)
        found_predecessor = False
        for detected_device in detected_devices:
            if detected_devices[detected_device][-1]==device:
                if search_result is not None:
                    detected_devices[detected_device].append(search_result)
                found_predecessor = True
        if not found_predecessor:
            i = len(detected_devices)
            detected_devices[i] = [device]
            if search_result is not None:
                detected_devices[i].append(search_result)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to open.", type=str, )
    parser.add_argument("-v", "--verbose", help="Print more information about devices.", action="store_true")
    parser.add_argument("-d", "--draw", help="Draw potential tracking devices chart.", action="store_true")
    parser.add_argument("-D", "--drawselected", help="Draw specific tracking devices chart.", type=str, required=False)
    parser.add_argument("-C", "--compare", help="Compare specific tracking devices chart.", type=int, required=False)
    parser.add_argument("--mintime", help="Minimal emission duration for a MAC address (in seconds).", type=int, required=False)
    parser.add_argument("--maxtime", help="Maximal emission duration for a MAC address (in seconds).", type=int, required=False)
    parser.add_argument("--adv", help="Mean advertisement period threshold.", type=int, required=False)
    parser.add_argument("--rssi", help="Mean RSSI threshold.", type=int, required=False)
    parser.add_argument("--savgolwindow", help="Savgol filter minimum window length.", type=int, required=False)
    parser.add_argument("--savgolpolynomial", help="Savgol filter polynomial.", type=int, required=False)
    parser.add_argument("--savgol", help="Enable Savgol filtering.", action="store_true", required=False)
    parser.add_argument("--nogaussian", help="Disable Gaussian filtering.", action="store_true", required=False)
    parser.add_argument("--gaussiansigma", help="Gaussian Sigma value.", type=int, required=False)
    args = parser.parse_args()
    if args.mintime != None:
        TIME_MIN_THRESHOLD = args.mintime
    if args.maxtime != None:
        TIME_MAX_THRESHOLD = args.maxtime
    if args.adv != None:
        MEAN_ADV_PERIOD_THRESHOLD = args.adv
    if args.rssi != None:
        MEAN_RSSI_THRESHOLD = args.rssi
    if args.savgolwindow != None:
        WINDOW_LENGTH = args.savgolwindow
    if args.savgolpolynomial != None:
        POLYORDER = args.savgolpolynomial
    SAVGOL_ENABLED = args.savgol
    if SAVGOL_ENABLED:
        if POLYORDER>=WINDOW_LENGTH:
            print("Savgol filter polynomial MUST be inferior to its window length.")
            exit(42)
    GAUSSIAN_ENABLED = (not args.nogaussian)
    if GAUSSIAN_ENABLED:
        if args.gaussiansigma!=None:
            GAUSSIAN_SIGMA = args.gaussiansigma
    input_measurements, raw_measurements = get_data(args.file)
    (TIME_MIN_GLOBAL, TIME_MAX_GLOBAL) = set_global_time_intervals(verbose=(args.verbose==True)) #weird!
    successors_quest()
    if (args.compare!=None):
        compare_metadata_between_ids(args.compare)
        exit(1)
    if (args.drawselected==None and args.draw==False):
        print_devices(verbose=(args.verbose==True))
    if args.drawselected!=None:
        selected = [ int(x) for x in args.drawselected.split(',') ]
        draw_devices(selected)
    elif args.draw==True:
        if len(select_devices())>0:
            draw_devices(select_devices())