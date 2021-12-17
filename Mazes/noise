import math, random

# Algorithmic structure based on https://en.wikipedia.org/wiki/Perlin_noise
class Perlin(object):
    # Takes in a list of octave scales and creates an object that takes in a
    # certain (x, y) value and create the value of the perlin noise at that.
    def __init__(self, octaves):
        self.octaves = octaves
        self.maxAmplitude = sum([amp for amp, _ in octaves])
        self.vDicts = [{} for i in range(len(octaves))]

    def __call__(self, x, y):
        result = 0
        for i in range(len(self.octaves)):
            # Extract the current scale and vectorDict
            amplitude, frequency = self.octaves[i]
            vectorDict = self.vDicts[i]
            # Calculate the x and y value when scaled by this octave
            oX, oY = x * frequency, y * frequency

            # Extract the bounding points of the tile containing the point
            x0, y0 = int(oX), int(oY)
            x1, y1 = x0 + 1, y0 + 1
            dx = oX - x0
            dy = oY - y0

            # Calculate the various dot products and interpolate the results
            n0 = self.dotGradient(x0, y0, oX, oY, vectorDict)
            n1 = self.dotGradient(x1, y0, oX, oY, vectorDict)
            ix0 = self.interpolate(n0, n1, dx)
            n0 = self.dotGradient(x0, y1, oX, oY, vectorDict)
            n1 = self.dotGradient(x1, y1, oX, oY, vectorDict)
            ix1 = self.interpolate(n0, n1, dx)
            value = self.interpolate(ix0, ix1, dy)
            # Add the interpolated value to the result, scaling by amplitude

            result += value * amplitude
        # Return the result normalized by the sum of all the amplitudes
        return result / self.maxAmplitude + 0.5

    # A cubic interpolation function
    def interpolate(self, a, b, w):
        return (b - a) * (3 - w * 2) * w ** 2 + a

    # Takes in a given (x, y) coordinate of a lattice point in a grid and returns
    # a random unit vector, which is stored in the vectorDict for future use.
    def getGradient(self, x, y, vectorDict):
        if (x, y) not in vectorDict:
            theta = random.random() * math.tau
            gradient = (math.cos(theta), math.sin(theta))
            vectorDict[(x, y)] = gradient
        return vectorDict[(x, y)]

    # Take the dot product of point of lattice point (x, y) to (x0, y0) and the
    # vector from lattice point (x, y)
    def dotGradient(self, x0, y0, x, y, vectorDict):
        gradient = self.getGradient(x0, y0, vectorDict)
        dx = x - x0
        dy = y - y0
        return dx * gradient[0] + dy * gradient[1]

# Algorithmic structure based on http://www.6by9.net/simplex-noise-for-c-and-python/
class Simplex(Perlin):
    def __call__(self, x, y):
        result = 0
        for i in range(len(self.octaves)):
            # Extract the current scale and vectorDict
            amplitude, frequency = self.octaves[i]
            vectorDict = self.vDicts[i]

            # Calculate the x and y value when scaled by this octave
            oX, oY = x * frequency, y * frequency

            # Use the 2D skew factor to calculate the triangle corner
            skewFactor = (3 ** 0.5 - 1) / 2
            s = (oX + oY) * skewFactor
            
            a0 = int(oX + s)
            b0 = int(oY + s)

            # Unskew to obtain the first displacement vector
            unskewFactor = (3 - 3 ** 0.5) / 6
            t = (a0 + b0) * unskewFactor
            x0 = oX - (a0 - t)
            y0 = oY - (b0 - t)

            # Determine if hte triangle is facing up or down, then calculate
            # the other two displacement vectors
            dx, dy = (1, 0) if x0 > y0 else (0, 1)
            x1 = x0 - dx + unskewFactor
            y1 = y0 - dy + unskewFactor
            x2 = x0 - 1 + 2 * unskewFactor
            y2 = y0 - 1 + 2 * unskewFactor

            # Extract the gradient vectors for all 3 lattice points
            v0 = self.getGradient(a0, b0, vectorDict)
            v1 = self.getGradient(a0+dx, b0+dy, vectorDict)
            v2 = self.getGradient(a0+1, b0+1, vectorDict)

            # For each of the 3 corners, use the dot products to extract
            # the noise values
            points = [(x0, y0, *v0), (x1, y1, *v1), (x2, y2, *v2)]
            noise = 0
            for x0, y0, x1, y1 in points:
                t = 0.5 - x0**2 - y0**2
                noise += 0 if t < 0 else t ** 4 * (x0 * x1 + y0 * y1)

            # Add the combined and normalized noise to the overall result,
            # weighted by the amplitude
            result += (0.5 + 35 * noise) * amplitude / self.maxAmplitude

        return result
