import asyncio
from bleak import BleakScanner
from datetime import datetime
import statistics
import json
import argparse
import base64
import io

device_pings = {}
start_time = datetime.now().timestamp()

class myencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, bytearray):
            return base64.b64encode(obj).decode('utf-8')
        else:
            return super().default(obj)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to open.", type=str, )
    parser.add_argument("time", help="Time of listening", type=int, )
    parser.add_argument("advs", help="Minimum advertising for a MAC addr", type=int, )
    parser.add_argument("-v", "--verbose", help="Print more information about devices.", action="store_true")
    args = parser.parse_args()
    stop_event = asyncio.Event()

    # TODO: add something that calls stop_event.set()
    def parse_adv(advertising):
        out = {}
        if advertising.local_name:
            out['local_name'] = advertising.local_name
        else:
            out['local_name'] = "N/A"
        if advertising.manufacturer_data:
            out['manufacturer_data'] = advertising.manufacturer_data
            for data in out['manufacturer_data']:
                out['manufacturer_data'][data] = base64.b64encode(out['manufacturer_data'][data])
        else:
            out['manufacturer_data'] = "N/A"
        if advertising.service_data:
            out['service_data'] = advertising.service_data
            for data in out['service_data']:
                out['service_data'][data] = base64.b64encode(out['service_data'][data])
        else:
            out['service_data'] = "N/A"
        if advertising.service_uuids:
            out['service_uuids'] = advertising.service_uuids
        else:
            out['service_uuids'] = "N/A"
        if advertising.tx_power:
            out['tx_power'] = advertising.tx_power
        else:
            out['tx_power'] = "N/A"
        if advertising.rssi:
            out['rssi'] = advertising.rssi
        else:
            out['rssi'] = "N/A"
        if advertising.platform_data:
            out['platform_data'] = advertising.platform_data
        else:
            out['platform_data'] = "N/A"
        return out
    def callback(device, advertising_data):
        if device.address not in device_pings:
            device_pings[device.address] = []
        output = {}
        device_pings[device.address].append({int(datetime.now().timestamp()): parse_adv(advertising_data)})

    async with BleakScanner(callback) as scanner:
        # Important! Wait for an event to trigger stop, otherwise scanner
        # will stop immediately.
        await asyncio.sleep(args.time)
        stop_event.set()
        with open(args.file, "w") as f:
            json.dump(device_pings, f, cls=myencoder, indent=1)
    # scanner stops when block exits

asyncio.run(main())
