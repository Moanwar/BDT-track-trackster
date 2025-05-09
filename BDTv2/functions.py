from numba import prange,njit
import numpy as np
import math 

@njit
def distWrap_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    out = []
    for i in range(len(otherTsEta)):
        deltaPhi = otherTsPhi[i] - refPhi
        deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
        distance = ((otherTsEta[i] - refEta) ** 2 + deltaPhi ** 2) ** 0.5
        out.append(distance)
    return np.array(out)

@njit
def flatten_numba(a):
    return [x[0] if len(x) else 0 for x in a]
@njit
def distpull_numba(refEta, refEtaErr ,refPhi, refPhiErr,
                   otherTsEta, otherTsEtaErr, otherTsPhi, otherTsPhiErr):
    out = []
    for i in range(len(otherTsEta)):
        deltaPhi  = otherTsPhi[i] - refPhi
        deltaPhi  = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
        delta_eta = (otherTsEta[i] - refEta) ** 2
        sigma_eta = np.sqrt(refEtaErr**2 + otherTsEtaErr[i]**2)
        sigma_phi = np.sqrt(refPhiErr**2 + otherTsPhiErr[i]**2)
        delta_eta_pull = delta_eta / sigma_eta
        delta_phi_pull = deltaPhi / sigma_phi
        delta_R_pull = np.sqrt(delta_eta_pull**2 + delta_phi_pull**2)
        out.append(delta_R_pull)
    return np.array(out)
@njit
def mtdValue_numba(refx, refy ,refz,
                   refBeta, reftime, reftimeErr,
                   otherTsX, otherTsY, otherTsZ,
                   otherTsTime, otherTsTimeErr):
    out = []
    for i in range(len(otherTsTime)):
        deltaSoverV = (
            np.sqrt(
                (otherTsX[i] - refx)**2 +
                (otherTsY[i] - refy)**2 +
                (otherTsZ[i] - refz)**2
            ) / (refBeta * 29.9792458)
        )
        
        deltaT = otherTsTime[i] - reftime
        sigma = np.sqrt(otherTsTimeErr[i]**2 + reftimeErr**2)
        MTDvalue = abs(deltaSoverV - deltaT) / sigma
        out.append(MTDvalue)
    return np.array(out)
