# Load configuration files within the folder
def read_config_file(filepath):
    config = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Remove unnecessary whitespace characters
            line = line.strip()
            # Skip empty or comment lines
            if not line or not line[0].isalpha():
                continue
            # Split the key and value
            key, value = line.split('=', 1)
            # Remove extra spaces from key and value, and convert values
            key = key.strip()
            value = value.strip()
            try:
                # Try to convert the value to a float or integer
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # If conversion fails, keep the value as a string
            config[key] = value
    return config

def read_station_file(filepath, conf):
    obs_sys={}
    obs_sys['sta_ids']=[]
    obs_sys['x']=[]
    obs_sys['y']=[]
    obs_sys['z']=[]
    xRef, yRef = conf['RefCoord'].split()[:2]
    xRef = float(xRef) * 1000
    yRef = float(yRef) * 1000
    with open(filepath, 'r') as file:
        for line in file:
            # Remove unnecessary whitespace characters
            line = line.strip()
            # Skip empty or comment lines
            if not line or not line[0].isalpha():
                continue
            # Split station ID and coordinates
            sta_id, x, y, z = line.split()
            # Add to list
            obs_sys['sta_ids'].append(sta_id)
            obs_sys['x'].append(round(float(x)-xRef, 3))
            obs_sys['y'].append(round(float(y)-yRef, 3))
            obs_sys['z'].append(round(float(z), 3))
    return obs_sys



if __name__ == '__main__':
    # Example usage
    conf_s1 = read_config_file('./config/conf_s1.txt')
    print(conf_s1)
    obs_sys = read_station_file('./config/station.txt', conf_s1)
    print(obs_sys)
