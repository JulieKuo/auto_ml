import subprocess
import hashlib
import json
import os
import platform


class LicenseChecker():
    def __init__(self, config_filepath):
        # Read config file
        self.config = self.read_json(config_filepath)
        # Get data
        self.registered = self.read_registered(self.config['register_file'])
        self.platform_id = self.config['platform_id']
        self.cpuid = self.get_cpu_id()

    def read_json(self, file_path):
        if not os.path.exists(file_path):
            raise(FileNotFoundError(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def read_registered(self, file_path):
        if not os.path.exists(file_path):
            raise(FileNotFoundError(file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            string = f.readline()
            f.close()
        return string

    def get_cpu_id(self):
        myos = platform.system()
        if myos == "Windows":
            self.cpuid = subprocess.check_output('wmic cpu get ProcessorId', text=True).replace(' ', '').split('\n')[2]
        elif myos == "Linux":
            self.cpuid = subprocess.check_output(["sudo", "dmidecode", "-t", "processor"], text=True).\
            split('\n')[10].replace('\tID:', '').replace(' ', '')
        else:
            print('OS value incorrect: [{}]'.format(myos))

        return self.cpuid

    def get_hashed_string(self):
        string = hashlib.md5((self.platform_id + '_' + self.cpuid).encode('utf-8')).hexdigest()
        return string

    def check_license(self):
        hashed_string = self.get_hashed_string()
        print(f"hashed_string = {hashed_string}")
        if hashed_string == self.registered:
            return 1
        else:
            return 0



def check_license(license_config_filepath):
    cl = LicenseChecker(license_config_filepath)

    # return 0 or 1
    check_result = cl.check_license()
    return check_result


if __name__ == "__main__":
    license_config_filepath = 'check_license.json'

    try:
        cl = LicenseChecker(license_config_filepath)
        check_result = cl.check_license()
        print(check_result)

        # # For testing, you can get registered value here
        print(cl.get_hashed_string())

        # print(cl.get_cpu_id())
    
    except FileNotFoundError as fe:
        print(f'file missing: {str(fe)}')
