#include <unordered_map>
#include <string>
#include <map>
#include <windows.h>
const float PI = 3.14159265358979;
// Global variables to assist in timing
double PCFreq = 0.0;
__int64 CounterStart = 0;

/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 *
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *   what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        exit(code);
    }
}

/* Checks the passed-in arguments for validity. */
void check_args(int argc, char **argv)
{
    if (argc != 3) // the first argument is the program name
    {
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Arguments: <threads per block> <path to .config folder> \n";
        exit(EXIT_FAILURE);
    }
}

// Initialize Windows-specific precise timing
void initTiming()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed! Timing routines won't work. \n");

    PCFreq = double(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}

// Get precise time
double preciseClock()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart) / PCFreq;
}

void PrintDeviceInfo()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of device(s): " << deviceCount << std::endl;
    if (deviceCount == 0)
    {
        std::cout << "There is no device supporting CUDA" << std::endl;
        return;
    }

    cudaDeviceProp info;
    for (int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&info, i);
        std::cout << "Device " << i << std::endl;
        std::cout << "    Name:                    " << std::string(info.name) << std::endl;
        std::cout << "    Glocbal memory:          " << info.totalGlobalMem / 1024.0 / 1024.0 << " MB" << std::endl;
        std::cout << "    Shared memory per block: " << info.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "    Warp size:               " << info.warpSize << std::endl;
        std::cout << "    Max thread per block:    " << info.maxThreadsPerBlock << std::endl;
        std::cout << "    Thread dimension limits: " << info.maxThreadsDim[0] << " x "
                  << info.maxThreadsDim[1] << " x "
                  << info.maxThreadsDim[2] << std::endl;
        std::cout << "    Max grid size:           " << info.maxGridSize[0] << " x "
                  << info.maxGridSize[1] << " x "
                  << info.maxGridSize[2] << std::endl;
        std::cout << "    Compute capability:      " << info.major << "." << info.minor << std::endl;
    }
}

inline int getIndex_staxyz(int sta, int x, int y, int z, int nx, int ny, int nz)
{
    return sta * nx * ny * nz + x * ny * nz + y * nz + z;
}
inline int getIndex_zstaxy(int sta, int x, int y, int z, int nsta, int nx, int ny)
{
    return z * nsta * nx * ny + sta * nx * ny + x * ny + y;
}

/**
 * calculates the properties of layers between a source and a
 * detector based on their altitude.
 */
void calculateLayerProperties(const float *boundaryAltitudes, const float *velocities,
                              int numLayers, float sourceAltitude, float detectorAltitude,
                              std::vector<float> &thicknesses,
                              std::vector<float> &velocitiesInRange)
{
    thicknesses.clear();
    velocitiesInRange.clear();

    for (int i = 0; i < numLayers; ++i)
    {
        float upperAltitude = boundaryAltitudes[i];
        float lowerAltitude = (i == numLayers - 1) ? -std::numeric_limits<float>::infinity() : boundaryAltitudes[i + 1];
        float layerVelocity = velocities[i];

        if (sourceAltitude < upperAltitude && detectorAltitude > lowerAltitude)
        //
        {
            float startAltitude = max(sourceAltitude, lowerAltitude);
            float endAltitude = min(detectorAltitude, upperAltitude);
            float thickness = endAltitude - startAltitude;

            if (thickness > 0)
            {
                thicknesses.push_back(thickness);
                velocitiesInRange.push_back(layerVelocity);
            }
        }
    }
}

class ObsSystem
{
public:
    int nsta;
    std::vector<std::string> sta_ids;
    std::vector<float> xValues, yValues, zValues;
    std::vector<int> roundedZValues;
    std::vector<int> classifiedAltitudes;
    ObsSystem(const std::string &filename, float xRef, float yRef, float zRef)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string id;
            float x_val, y_val, z_val;

            if (ss >> id >> x_val >> y_val >> z_val)
            {
                sta_ids.push_back(id);
                xValues.push_back(x_val - xRef);
                yValues.push_back(y_val - yRef);
                zValues.push_back(z_val);
            }
            else
            {
                std::cerr << "Error reading data from line: " << line << std::endl;
            }
        }

        file.close();
        nsta = sta_ids.size();
        if (nsta == 0)
        {
            std::cerr << "No stations found in file: " << filename << std::endl;
            return;
        }
        classifyStationsByAltitude();
    }

    ~ObsSystem()
    {
    }

    void printStations() const
    {
        std::cout << "[ Station Information ]" << std::endl;
        std::cout << "Number of Stations: " << nsta << std::endl;
        for (int i = 0; i < nsta; ++i)
            std::cout << "Station ID: " << sta_ids[i]
                      << ", X: " << xValues[i]
                      << ", Y: " << yValues[i]
                      << ", Z: " << zValues[i]
                      << std::endl;
    }

    void calculateEpicentralDistances(const std::vector<float> &gridX, const std::vector<float> &gridY, float *distances) const
    {
        // shape of distances is (nsta, gridX.size(), gridY.size())
        size_t index = 0;
        for (int j = 0; j < nsta; ++j)
        {
            for (float x : gridX)
            {
                for (float y : gridY)
                {
                    float dx = x - xValues[j];
                    float dy = y - yValues[j];
                    distances[index] = std::sqrt(dx * dx + dy * dy);
                    ++index;
                }
            }
        }
    }

    void classifyStationsByAltitude()
    {
        std::map<int, int> altitudeCount;

        for (int i = 0; i < nsta; ++i)
        {
            int roundedZ = static_cast<int>(std::round(zValues[i]));
            altitudeCount[roundedZ]++;
            roundedZValues.push_back(roundedZ);
        }

        // std::cout << "[ Altitude classification ]" << std::endl;
        // std::cout << "Number of Stations: " << nsta << std::endl;
        // std::cout << "Total number of altitude classes: " << altitudeCount.size() << std::endl
        //           << std::endl;
        for (const auto &entry : altitudeCount)
        {
            classifiedAltitudes.emplace_back(entry.first);
            // std::cout << "Altitude " << entry.first << "m: "
            //           << entry.second << " stations" << std::endl;
        }
    }
};

class Grid
{
public:
    float xStart, xCnt, xInterval, yStart, yCnt, yInterval, zStart, zCnt, zInterval;
    float xRef, yRef, zRef;
    // ew:x ns:y
    std::vector<float> xValues, yValues, zValues;
    unsigned int nx, ny, nz, ngrid;

    Grid(const std::string &filename)
    {
        xStart = xCnt = xInterval = yStart = yCnt = yInterval = zStart = zCnt = zInterval = 0.0f;

        std::map<std::string, std::string> config;
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                config[key] = value;
            }
        }
        file.close();
        xCnt = std::stof(config["SearchSizeX"]);
        yCnt = std::stof(config["SearchSizeY"]);
        zCnt = std::stof(config["SearchSizeZ"]);
        xStart = std::stof(config["SearchOriginX"]) * 1000;
        yStart = std::stof(config["SearchOriginY"]) * 1000;
        zStart = std::stof(config["SearchOriginZ"]) * 1000;
        xInterval = yInterval = std::stof(config["GridSpacingX"]) * 1000;
        zInterval = std::stof(config["GridSpacingZ"]) * 1000;
        std::stringstream ss(config["RefCoord"]);
        ss >> xRef >> yRef >> zRef;
        xRef *= 1000;
        yRef *= 1000;
        zRef *= 1000;
        calcXYZValues();
    }

    void calcXYZValues()
    {
        for (int i = 0; i < xCnt; i++)
        {
            xValues.push_back(xStart + i * xInterval);
        }
        for (int i = 0; i < yCnt; i++)
        {
            yValues.push_back(yStart + i * yInterval);
        }
        for (int i = 0; i < zCnt; i++)
        {
            zValues.push_back(zStart + i * zInterval);
        }
        nx = xValues.size();
        ny = yValues.size();
        nz = zValues.size();
        ngrid = nx * ny * nz;
        // std::cout << "Length of grid.zValues: " << zValues.size() << std::endl;
    }
    void printConf()
    {
        std::cout << "[ Grid Config ]" << std::endl;
        std::cout << "xStart: " << xStart << std::endl;
        std::cout << "yStart: " << yStart << std::endl;
        std::cout << "zStart: " << zStart << std::endl;
        std::cout << "xCnt: " << xCnt << std::endl;
        std::cout << "yCnt: " << yCnt << std::endl;
        std::cout << "zCnt: " << zCnt << std::endl;
        std::cout << "xInterval: " << xInterval << std::endl;
        std::cout << "yInterval: " << yInterval << std::endl;
        std::cout << "zInterval: " << zInterval << std::endl;
        std::cout << "xRef: " << xRef << std::endl;
        std::cout << "yRef: " << yRef << std::endl;
        std::cout << "zRef: " << zRef << std::endl;
    }
};

void saveData(std::string attr, std::vector<std::string> sta_ids, const float *data, const int ngrid)
{
    // Check if path exists
    std::string path = "./LOC/" + attr;
    DWORD dwAttrib = GetFileAttributes(path.c_str());
    if (dwAttrib == INVALID_FILE_ATTRIBUTES)
    {
        // Directory does not exist, create directory
        if (CreateDirectory(path.c_str(), NULL))
        {
            std::cout << "Directory created: " << path << std::endl;
        }
        else
        {
            std::cerr << "Failed to create directory: " << path << std::endl;
            return;
        }
    }
    std::cout << "saving data..." << std::endl;
    for (int i = 0; i < sta_ids.size(); i++)
    {
        std::string filename = "./LOC/" + attr + "/layer.P." + sta_ids[i].substr(2) + "." + attr + ".buf";
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        // Check if file opened successfully
        if (!file)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            continue; // If file opening fails, skip current station
        }
        file.write((char *)data + i * ngrid * sizeof(float), ngrid * sizeof(float));
        file.close();
    }
}

std::string toUpper(const std::string &str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

void saveDataInfo(std::string attr, ObsSystem osys, Grid g)
{
    // Check if path exists
    std::string path = "./LOC/" + attr;
    DWORD dwAttrib = GetFileAttributes(path.c_str());
    if (dwAttrib == INVALID_FILE_ATTRIBUTES)
    {
        // Directory does not exist, create directory
        if (CreateDirectory(path.c_str(), NULL))
        {
            std::cout << "Directory created: " << path << std::endl;
        }
        else
        {
            std::cerr << "Failed to create directory: " << path << std::endl;
            return;
        }
    }
    for (int i = 0; i < osys.sta_ids.size(); i++)
    {
        std::string filename = "./LOC/" + attr + "/layer.P." + osys.sta_ids[i].substr(2) + "." + attr + ".hdr";
        std::ofstream file(filename, std::ios::out);
        // Check if file opened successfully
        if (!file)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            continue; // If file opening fails, skip current station
        }
        /**
         * Format:
         * xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType
         * staid stax stay staz
         */
        file << g.nx << " " << g.ny << " " << g.nz << " ";
        file << g.xStart / 1000 << " " << g.yStart / 1000 << " " << g.zStart / 1000 << " ";
        file << g.xInterval / 1000 << " " << g.yInterval / 1000 << " " << g.zInterval / 1000 << " ";
        file << toUpper(attr) << std::endl;
        file << osys.sta_ids[i].substr(2) << " ";
        file << osys.xValues[i] / 1000 << " " << osys.yValues[i] / 1000 << " " << osys.zValues[i] / 1000 << std::endl;
    }
}
