"""
Python implementation of Falcon:
https://falcon-sign.info/.
"""
from common import q
from numpy import set_printoptions, ones, zeros, random
from math import sqrt, exp, floor, ceil, log
from fft import fft, ifft, sub, neg, add_fft, mul_fft, add, sub_fft
from ntt import sub_zq, mul_zq, div_zq
from ffsampling import gram, ffldl_fft, ffsampling_fft, ffsampling_round
from ntrugen import ntru_gen
from encoding import compress, decompress
# https://pycryptodome.readthedocs.io/en/latest/src/hash/shake256.html
from Crypto.Hash import SHAKE256
# Randomness
from os import urandom
from rng import ChaCha20
# For debugging purposes
import sys
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.

from random import uniform
from copy import deepcopy
from numpy import round as np_round

from timeit import default_timer as timer


set_printoptions(linewidth=200, precision=5, suppress=True)

logn = {
    2: 1,
    4: 2,
    8: 3,
    16: 4,
    32: 5,
    64: 6,
    128: 7,
    256: 8,
    512: 9,
    1024: 10
}


# Bytelength of the signing salt and header
HEAD_LEN = 1
SALT_LEN = 40
SEED_LEN = 56


# Parameter sets for Falcon:
# - n is the dimension/degree of the cyclotomic ring
# - sigma is the std. dev. of signatures (Gaussians over a lattice)
# - sigmin is a lower bounds on the std. dev. of each Gaussian over Z
# - sigbound is the upper bound on ||s0||^2 + ||s1||^2
# - sig_bytelen is the bytelength of signatures
Params = {
    # FalconParam(2, 2)
    2: {
        "n": 2,
        "sigma": 144.81253976308423,
        "sigmin": 1.1165085072329104,
        "sig_bound": 101498,
        "sig_bytelen": 44,
    },
    # FalconParam(4, 2)
    4: {
        "n": 4,
        "sigma": 146.83798833523608,
        "sigmin": 1.1321247692325274,
        "sig_bound": 208714,
        "sig_bytelen": 47,
    },
    # FalconParam(8, 2)
    8: {
        "n": 8,
        "sigma": 148.83587593064718,
        "sigmin": 1.147528535373367,
        "sig_bound": 428865,
        "sig_bytelen": 52,
    },
    # FalconParam(16, 4)
    16: {
        "n": 16,
        "sigma": 151.78340713845503,
        "sigmin": 1.170254078853483,
        "sig_bound": 892039,
        "sig_bytelen": 63,
    },
    # FalconParam(32, 8)
    32: {
        "n": 32,
        "sigma": 154.6747794602761,
        "sigmin": 1.1925466358390344,
        "sig_bound": 1852696,
        "sig_bytelen": 82,
    },
    # FalconParam(64, 16)
    64: {
        "n": 64,
        "sigma": 157.51308555044122,
        "sigmin": 1.2144300507766141,
        "sig_bound": 3842630,
        "sig_bytelen": 122,
    },
    # FalconParam(128, 32)
    128: {
        "n": 128,
        "sigma": 160.30114421975344,
        "sigmin": 1.235926056771981,
        "sig_bound": 7959734,
        "sig_bytelen": 200,
    },
    # FalconParam(256, 64)
    256: {
        "n": 256,
        "sigma": 163.04153322607107,
        "sigmin": 1.2570545284063217,
        "sig_bound": 16468416,
        "sig_bytelen": 356,
    },
    # FalconParam(512, 128)
    512: {
        "n": 512,
        "sigma": 165.7366171829776,
        "sigmin": 1.2778336969128337,
        "sig_bound": 34034726,
        "sig_bytelen": 666,
    },
    # FalconParam(1024, 256)
    1024: {
        "n": 1024,
        "sigma": 168.38857144654395,
        "sigmin": 1.298280334344292,
        "sig_bound": 70265242,
        "sig_bytelen": 1280,
    },
}


def print_tree(tree, pref=""):
    """
    Display a LDL tree in a readable form.

    Args:
        T: a LDL tree

    Format: coefficient or fft
    """
    leaf = "|_____> "
    top = "|_______"
    son1 = "|       "
    son2 = "        "
    width = len(top)

    a = ""
    if len(tree) == 3:
        if (pref == ""):
            a += pref + str(tree[0]) + "\n"
        else:
            a += pref[:-width] + top + str(tree[0]) + "\n"
        a += print_tree(tree[1], pref + son1)
        a += print_tree(tree[2], pref + son2)
        return a

    else:
        return (pref[:-width] + leaf + str(tree) + "\n")


def normalize_tree(tree, sigma):
    """
    Normalize leaves of a LDL tree (from values ||b_i||**2 to sigma/||b_i||).

    Args:
        T: a LDL tree
        sigma: a standard deviation

    Format: coefficient or fft
    """
    if len(tree) == 3:
        normalize_tree(tree[1], sigma)
        normalize_tree(tree[2], sigma)
    else:
        tree[0] = sigma / sqrt(tree[0].real)
        tree[1] = 0


class PublicKey:
    """
    This class contains methods for performing public key operations in Falcon.
    """

    def __init__(self, sk):
        """Initialize a public key."""
        self.n = sk.n
        self.h = sk.h
        self.hash_to_point = sk.hash_to_point
        self.signature_bound = sk.signature_bound
        self.verify = sk.verify

    def __repr__(self):
        """Print the object in readable form."""
        rep = "Public for n = {n}:\n\n".format(n=self.n)
        rep += "h = {h}\n\n".format(h=self.h)
        return rep


class SecretKey:
    """
    This class contains methods for performing
    secret key operations (and also public key operations) in Falcon.

    One can:
    - initialize a secret key for:
        - n = 128, 256, 512, 1024,
        - phi = x ** n + 1,
        - q = 12 * 1024 + 1
    - find a preimage t of a point c (both in ( Z[x] mod (Phi,q) )**2 ) such that t*B0 = c
    - hash a message to a point of Z[x] mod (Phi,q)
    - sign a message
    - verify the signature of a message
    """

    def __init__(self, n, polys=None):
        """Initialize a secret key."""
        # Public parameters
        self.n = n
        #TODO: change sigma and signature_bound
        self.sigma = Params[n]["sigma"]
        self.sigmin = Params[n]["sigmin"]
        self.sigma_old = None
        self.signature_bound = floor(Params[n]["sig_bound"])
        self.sig_bytelen = Params[n]["sig_bytelen"]

        # Compute NTRU polynomials f, g, F, G verifying fG - gF = q mod Phi
        if polys is None:
            self.f, self.g, self.F, self.G = ntru_gen(n)
        else:
            [f, g, F, G] = polys
            assert all((len(poly) == n) for poly in [f, g, F, G])
            self.f = f[:]
            self.g = g[:]
            self.F = F[:]
            self.G = G[:]

        # From f, g, F, G, compute the basis B0 of a NTRU lattice
        # as well as its Gram matrix and their fft's.
        B0 = [[self.g, neg(self.f)], [self.G, neg(self.F)]]
        G0 = gram(B0)
        self.B0_fft = [[fft(elt) for elt in row] for row in B0]
        G0_fft = [[fft(elt) for elt in row] for row in G0]

        self.T_fft = ffldl_fft(G0_fft)

        '''
        store original T_fft
        '''
        self.orig_T_fft = deepcopy(self.T_fft)

        # Normalize Falcon tree
        normalize_tree(self.T_fft, self.sigma)

        # The public key is a polynomial such that h*f = g mod (Phi,q)
        self.h = div_zq(self.g, self.f)

    def __repr__(self, verbose=False):
        """Print the object in readable form."""
        rep = "Private key for n = {n}:\n\n".format(n=self.n)
        rep += "f = {f}\n\n".format(f=self.f)
        rep += "g = {g}\n\n".format(g=self.g)
        rep += "F = {F}\n\n".format(F=self.F)
        rep += "G = {G}\n\n".format(G=self.G)
        if verbose:
            rep += "\nFFT tree\n"
            rep += print_tree(self.T_fft, pref="")
        return rep

    def hash_to_point(self, message, salt):
        """
        Hash a message to a point in Z[x] mod(Phi, q).
        Inspired by the Parse function from NewHope.
        """
        n = self.n
        if q > (1 << 16):
            raise ValueError("The modulus is too large")

        k = (1 << 16) // q
        # Create a SHAKE object and hash the salt and message.
        shake = SHAKE256.new()
        shake.update(salt)
        shake.update(message)
        # Output pseudorandom bytes and map them to coefficients.
        hashed = [0 for i in range(n)]
        i = 0
        j = 0
        while i < n:
            # Takes 2 bytes, transform them in a 16 bits integer
            twobytes = shake.read(2)
            elt = (twobytes[0] << 8) + twobytes[1]  # This breaks in Python 2.x
            # Implicit rejection sampling
            if elt < k * q:
                hashed[i] = elt % q
                i += 1
            j += 1
        return hashed

    def sample_preimage(self, point, type_in, sigma_new, i_mix_sym, overwrite, h=220, k_hmc=3, seed=None):
        """
        Sample a short vector s such that s[0] + s[1] * h = point.
        """
        [[a, b], [c, d]] = self.B0_fft

        # We compute a vector t_fft such that:
        #     (fft(point), fft(0)) * B0_fft = t_fft
        # Because fft(0) = 0 and the inverse of B has a very specific form,
        # we can do several optimizations.

        point_fft = fft(point)

        t0_fft = [(point_fft[i] * d[i]) / q for i in range(self.n)]
        t1_fft = [(-point_fft[i] * b[i]) / q for i in range(self.n)]
        t_fft = [t0_fft, t1_fft]

        # h is the step size for MALA
        h_factor = h / (self.sigma_old**2)

        '''
        MCMC sampling
        '''
        # Get initial state z_0 and i_mix
        # If no MCMC sampling, this is the solution for original FALCON
        i_mix = None
        if seed is None:
            # If no seed is defined, use urandom as the pseudo-random source.
            z_0, sum_log_prob_0, i_mix = ffsampling_fft(t_fft, self.T_fft, self.sigmin, 0, 1, False, urandom)

        else:
            # If a seed is defined, initialize a ChaCha20 PRG
            # that is used to generate pseudo-randomness.
            chacha_prng = ChaCha20(seed)
            z_0, sum_log_prob_0, i_mix = ffsampling_fft(t_fft, self.T_fft, self.sigmin, 0, 1, False,
                                   chacha_prng.randombytes)

        '''
        # When initiating with round(t_fft) instead of ffsampling for symmetric MCMC
        z_round = np_round(t_fft)
        z_test, _, _ = ffsampling_fft(t_fft, self.T_fft, self.sigmin, 0, 1, urandom)

        v0_test, v1_test = self.calc_v(z_test)
        s_test = [sub(point, v0_test), neg(v1_test)]
        test_norm = self.calc_norm(s_test)

        v0_round, v1_round = self.calc_v(z_round)
        s_round = [sub(point, v0_round), neg(v1_round)]
        round_norm = self.calc_norm(s_round) 

        print("z_0: ", og_squared_norm)
        print("z_round: ", round_norm)
        print("z_test: ", test_norm)
        '''

        if self.sigma_old is not None and sigma_new is not None and type_in != 'imhk':
            # normalize tree
            self.sigma = float(sigma_new)
            self.T_fft = deepcopy(self.orig_T_fft)
            normalize_tree(self.T_fft, float(sigma_new))
            # factor for MALK and HMCK
            h_factor = h / (self.sigma_old ** 2)

        ''''Testing
        print("imix: ", i_mix)
        v0_og, v1_og = self.calc_v(z_0)
        s_og = [sub(point, v0_og), neg(v1_og)]
        og_squared_norm = self.calc_norm(s_og)
        og_sum_log_prob = sum_log_prob_0
        num_moves = 0
        num_good_moves = 0
        progress_stats_1 = [og_squared_norm]
        progress_stats_2 = [og_squared_norm]
        End Test'''

        if type_in == 'imhk':
            if overwrite:
                i_mix = 1
            for i in range(ceil(i_mix) - 1):
                if seed is None:
                    # If no seed is defined, use urandom as the pseudo-random source.
                    z_fft, sum_log_prob_1, _ = ffsampling_fft(t_fft, self.T_fft, self.sigmin, 0, 1, True, urandom)

                else:
                    # If a seed is defined, initialize a ChaCha20 PRG
                    # that is used to generate pseudo-randomness.
                    chacha_prng = ChaCha20(seed)
                    z_fft, sum_log_prob_1, _ = ffsampling_fft(t_fft, self.T_fft, self.sigmin, 0, 1, True,
                                           chacha_prng.randombytes)

                old_new_ratio = sum_log_prob_1 - sum_log_prob_0
                acceptance_ratio = min(0, old_new_ratio)
                u = uniform(0, 1)

                # cannot be 0 due to log
                while u == 0:
                    u = uniform(0, 1)

                #print("[", i+1, "]: new_sum: ", sum_log_prob_1, ", old_sum: ", sum_log_prob_0)

                '''logging
                v0_b, v1_b = self.calc_v(z_0)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_1.append(squared_norm_b)

                v0_b, v1_b = self.calc_v(z_fft)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_2.append(squared_norm_b)
                end logging'''

                if log(u) <= acceptance_ratio:
                    #print("\naccepted -- ", "ratio: ", acceptance_ratio, ", log(u): ", log(u), "\n")
                    z_0 = z_fft
                    sum_log_prob_0 = sum_log_prob_1

                    '''testing
                    num_moves += 1
                    if old_new_ratio > 0:
                        num_good_moves += 1
                    end testing'''

        elif type_in == 'smk':
            i_mix = i_mix_sym

            for i in range(i_mix):
                if seed is None:
                    # If no seed is defined, use urandom as the pseudo-random source.
                    z_fft, sum_log_prob, _ = ffsampling_fft(z_0, self.T_fft, self.sigmin, 0, 1, True, urandom)

                else:
                    # If a seed is defined, initialize a ChaCha20 PRG
                    # that is used to generate pseudo-randomness.
                    chacha_prng = ChaCha20(seed)
                    z_fft, sum_log_prob, _ = ffsampling_fft(z_0, self.T_fft, self.sigmin, 0, 1, True,
                                           chacha_prng.randombytes)

                v0_new, v1_new = self.calc_v(z_fft)
                v0_old, v1_old = self.calc_v(z_0)

                # The difference s = (point, 0) - v is such that:
                #     s is short
                #     s[0] + s[1] * h = point
                s_new = [sub(point, v0_new), neg(v1_new)]
                s_old = [sub(point, v0_old), neg(v1_old)]
                new_squared_norm = self.calc_norm(s_new)
                old_squared_norm = self.calc_norm(s_old)

                old_new_ratio = exp((1 / (2 * (self.sigma_old ** 2))) * (old_squared_norm - new_squared_norm))
                acceptance_ratio = min(1, old_new_ratio)
                u = uniform(0, 1)
                #print("[", i+1, "]: new_squared_norm: ", new_squared_norm, ", old_squared_norm: ", old_squared_norm)

                '''logging
                v0_b, v1_b = self.calc_v(z_0)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_1.append(squared_norm_b)

                v0_b, v1_b = self.calc_v(z_fft)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_2.append(squared_norm_b)
                end logging'''

                if u <= acceptance_ratio:
                    #print("\naccepted -- ", "ratio: ", acceptance_ratio, ", u: ", u, "\n")
                    z_0 = z_fft

                    '''testing
                    num_moves += 1
                    if old_new_ratio > 1:
                        num_good_moves += 1
                    end testing'''

        elif type_in == 'malk':
            i_mix = i_mix_sym
            s_new_1 = [0, 0]
            s_new_2 = [0, 0]
            s_new_B = [0, 0]
            s_old_B = [0, 0]

            # calculate the first shifted step
            z_c = [0, 0]
            s_new_1[0] = sub_fft(z_0[0], t_fft[0])
            s_new_1[1] = sub_fft(z_0[1], t_fft[1])

            s_new_2[0] = [x * h_factor for x in s_new_1[0]]
            s_new_2[1] = [x * h_factor for x in s_new_1[1]]

            z_c[0] = sub(z_0[0], s_new_2[0])
            z_c[1] = sub(z_0[1], s_new_2[1])

            for i in range(i_mix):
                if seed is None:
                    # If no seed is defined, use urandom as the pseudo-random source.
                    z_fft, sum_log_prob_1, _ = ffsampling_fft(z_c, self.T_fft, self.sigmin, 0, 1, True, urandom)

                else:
                    # If a seed is defined, initialize a ChaCha20 PRG
                    # that is used to generate pseudo-randomness.
                    chacha_prng = ChaCha20(seed)
                    z_fft, sum_log_prob_1, _ = ffsampling_fft(z_c, self.T_fft, self.sigmin, 0, 1, True,
                                                            chacha_prng.randombytes)
                # By
                v0_new, v1_new = self.calc_v(z_fft)
                # Bx
                v0_old, v1_old = self.calc_v(z_0)

                # The difference s = (hashed, 0) - v is such that:
                #     s is short
                #     s[0] + s[1] * h = hashed
                # By-c
                s_new = [sub(point, v0_new), neg(v1_new)]
                # Bx-c
                s_old = [sub(point, v0_old), neg(v1_old)]
                new_squared_norm = self.calc_norm(s_new)
                old_squared_norm = self.calc_norm(s_old)

                # mala specific
                # h/sigma^2*(By-c)
                s_new_B[0] = [x * h_factor for x in s_new[0]]
                s_new_B[1] = [x * h_factor for x in s_new[1]]
                # h/sigma^2*(Bx-c)
                s_old_B[0] = [x * h_factor for x in s_old[0]]
                s_old_B[1] = [x * h_factor for x in s_old[1]]

                # Bx-By
                b_old_new_0 = sub(v0_old, v0_new)
                b_old_new_1 = sub(v1_old, v1_new)

                # By-Bx
                b_new_old_0 = sub(v0_new, v0_old)
                b_new_old_1 = sub(v1_new, v1_old)

                # (Bx-By)+h/sigma^2*(By-c)
                term_full_o_n = [add(b_old_new_0, s_new_B[0]), add(b_old_new_1, s_new_B[1])]
                # (By-Bx)+h/sigma^2*(Bx-c)
                term_full_n_o = [add(b_new_old_0, s_old_B[0]), add(b_new_old_1, s_old_B[1])]

                o_n_squared_norm_mala = self.calc_norm(term_full_o_n)
                n_o_squared_norm_mala = self.calc_norm(term_full_n_o)

                term_1 = new_squared_norm - old_squared_norm
                term_2 = o_n_squared_norm_mala - n_o_squared_norm_mala

                # since we use a two step sigma process
                if i == 0:
                    sum_log_prob_0 = sum_log_prob_1
                term_3 = sum_log_prob_1 - sum_log_prob_0

                # taking the log in order not to get an overflow
                old_new_ratio_1 = -(1 / (2 * (self.sigma_old ** 2))) * (term_1)
                old_new_ratio_2 = -(1 / (2 * (self.sigma ** 2))) * (term_2)
                acceptance_ratio = min(0, old_new_ratio_1+old_new_ratio_2+term_3)
                u = uniform(0, 1)
                while u == 0:
                    u = uniform(0, 1)

                #print("[", i + 1, "]: new_squared_norm: ", new_squared_norm, ", old_squared_norm: ", old_squared_norm,
                #      "accept ratio: ", acceptance_ratio, "log(u): ", log(u))

                '''logging
                v0_b, v1_b = self.calc_v(z_0)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_1.append(squared_norm_b)

                v0_b, v1_b = self.calc_v(z_fft)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_2.append(squared_norm_b)
                end logging'''

                if log(u) <= acceptance_ratio:
                    # print("\naccepted -- ", "ratio: ", acceptance_ratio, ", log(u): ", log(u), "\n")
                    # z_fft - h/sigma^2*(z_fft-t_fft)
                    s_new_1[0] = sub_fft(z_fft[0], t_fft[0])
                    s_new_1[1] = sub_fft(z_fft[1], t_fft[1])

                    s_new_2[0] = [x * h_factor for x in s_new_1[0]]
                    s_new_2[1] = [x * h_factor for x in s_new_1[1]]

                    z_c[0] = sub(z_fft[0], s_new_2[0])
                    z_c[1] = sub(z_fft[1], s_new_2[1])

                    # Updating State and sum_log_prob
                    z_0 = z_fft
                    sum_log_prob_0 = sum_log_prob_1

                    '''testing
                    num_moves += 1
                    if old_new_ratio_1 > 0:
                        num_good_moves += 1
                    end testing'''

        elif type_in == 'hmck':
            i_mix = i_mix_sym
            qk_q0 = [0, 0]
            q0_qk = [0, 0]
            p_zB = [0, 0]
            p_0B = [0, 0]

            for i in range(i_mix):

                # Approximate Hamilton dynamics
                # choose random initial momentum p
                p_0 = []
                for k in range(len(point)):
                    p_0.append(random.normal(0, 1, 1)[0])
                p_fft = fft(p_0)

                # use 2*q as divisor to get similar results as mala i.e. to make the momentum less aggressive
                p_fft_0 = [(p_fft[i] * d[i]) / (q) for i in range(self.n)]
                p_fft_1 = [(-p_fft[i] * b[i]) / (q) for i in range(self.n)]
                p_0 = [p_fft_0, p_fft_1]

                q_k = z_0
                p_k = p_0

                for j in range(k_hmc):
                    q_k, p_k = self.leapfrog(q_k, p_k, h, self.sigma_old, t_fft)

                if seed is None:
                    # If no seed is defined, use urandom as the pseudo-random source.
                    z_fft, sum_log_prob_1, _ = ffsampling_fft(q_k, self.T_fft, self.sigmin, 0, 1, True, urandom)

                else:
                    # If a seed is defined, initialize a ChaCha20 PRG
                    # that is used to generate pseudo-randomness.
                    chacha_prng = ChaCha20(seed)
                    z_fft, sum_log_prob_1, _ = ffsampling_fft(q_k, self.T_fft, self.sigmin, 0, 1, True,
                                                            chacha_prng.randombytes)

                # Take the reverse steps (for acceptance ratio calculation)
                p_fft[0] = neg(p_k[0])
                p_fft[1] = neg(p_k[1])
                z_fft_k = z_fft

                for j in range(k_hmc):
                    z_fft_k, p_fft = self.leapfrog(z_fft_k, p_fft, h, self.sigma_old, t_fft)

                # Bq'K
                v0_new, v1_new = self.calc_v(z_fft)
                # Bq0
                v0_old, v1_old = self.calc_v(z_0)
                # BqK
                v0_old_k, v1_old_k = self.calc_v(q_k)
                # Bq'KK
                v0_new_k, v1_new_k = self.calc_v(z_fft_k)
                # Bp_z
                p_zB[0], p_zB[1] = self.calc_v(p_k)
                # Bp_0
                p_0B[0], p_0B[1] = self.calc_v(p_0)

                # H(q,p) = 1/2(1/sigma^2||Bq-c||+||Bp||^2)
                # Bqk-c
                s_new = [sub(point, v0_new), neg(v1_new)]
                new_squared_norm = self.calc_norm(s_new)

                # Bq0-c
                s_old = [sub(point, v0_old), neg(v1_old)]
                old_squared_norm = self.calc_norm(s_old)

                s_new_B = 1/(self.sigma_old**2) * new_squared_norm
                s_old_B = 1/(self.sigma_old**2) * old_squared_norm

                # ||Bp||^2
                p_k_squared_norm = self.calc_norm(p_zB)
                p_0_squared_norm = self.calc_norm(p_0B)

                h_q_k = 1/2*(s_new_B - p_k_squared_norm)
                h_q_0 = 1/2*(s_old_B - p_0_squared_norm)

                term_1 = h_q_0 - h_q_k

                # Term 2:
                # Bq'k-Bq_k (bottom) resp. Bq_0-Bq'_kk (top)
                qk_q0[0] = sub(v0_new, v0_old_k)
                qk_q0[1] = sub(v1_new, v1_old_k)

                q0_qk[0] = sub(v0_old, v0_new_k)
                q0_qk[1] = sub(v1_old, v1_new_k)

                qk_q0_squared_norm = self.calc_norm(qk_q0)
                q0_qk_squared_norm = self.calc_norm(q0_qk)

                term_2 = q0_qk_squared_norm - qk_q0_squared_norm

                # since we use a two step sigma process
                if i == 0:
                    sum_log_prob_0 = sum_log_prob_1
                term_3 = sum_log_prob_1 - sum_log_prob_0

                # taking the log in order not to get an overflow
                term_2 = -(1 / (2 * (sigma_new ** 2))) * term_2
                acceptance_ratio = min(0, term_1 + term_2 + term_3)
                u = uniform(0, 1)
                while u == 0:
                    u = uniform(0, 1)

                #print("[", i + 1, "]: new_squared_norm: ", new_squared_norm, ", old_squared_norm: ", old_squared_norm,
                #      "accept ratio: ", acceptance_ratio, "log(u): ", log(u))

                '''logging
                v0_b, v1_b = self.calc_v(z_0)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_1.append(squared_norm_b)

                v0_b, v1_b = self.calc_v(z_fft)
                s_b = [sub(point, v0_b), neg(v1_b)]
                squared_norm_b = self.calc_norm(s_b)
                progress_stats_2.append(squared_norm_b)
                end logging'''


                if log(u) <= acceptance_ratio:
                    #print("\naccepted -- ", "ratio: ", acceptance_ratio, ", log(u): ", log(u), "\n")

                    # Updating State and sum_log_prob
                    z_0 = z_fft
                    sum_log_prob_0 = sum_log_prob_1

                    '''testing
                    new_squared_norm = self.calc_norm(s_new)
                    old_squared_norm = self.calc_norm(s_old)

                    old_new_ratio = (old_squared_norm - new_squared_norm)
                    num_moves += 1
                    if old_new_ratio > 0:
                        num_good_moves += 1
                    end testing'''

        v0, v1 = self.calc_v(z_0)
        s = [sub(point, v0), neg(v1)]

        '''Testing
        final_squared_norm = self.calc_norm(s)
        print("\nOriginal squared norm: ", og_squared_norm, "; Final squared norm: ", final_squared_norm, "\n")
        print("\nOriginal sum log prob: ", og_sum_log_prob, "; Final sum log prob: ", sum_log_prob_0, "\n")
        print("\nNumber of Markov moves: ", num_moves, "\n")
        print("\nNumber of 'Good' Markov moves: ", num_good_moves, "\n")
        print("progress_stats1 (accepted): ", progress_stats_1)
        print("progress_stats2 (proposed): ", progress_stats_2)
        end testing'''

        return s

    def leapfrog(self, q_hmc, p, h, sig, t_fft):
        '''pk = p - h/(2*sig**2)*(q - c)
        qk = q + h*pk
        pk = pk - h/(2*sig**2)*(qk - c)'''
        pk = [0, 0]
        qk = [0, 0]
        hpk = [0, 0]
        q_c = [0, 0]
        q_c_2 = [0, 0]
        qk_c = [0, 0]
        qk_c_2 = [0, 0]
        step = h/(2*sig**2)
        # pk
        q_c[0] = sub_fft(q_hmc[0], t_fft[0])
        q_c[1] = sub_fft(q_hmc[1], t_fft[1])

        q_c_2[0] = [x * step for x in q_c[0]]
        q_c_2[1] = [x * step for x in q_c[1]]

        pk[0] = sub_fft(p[0], q_c_2[0])
        pk[1] = sub_fft(p[1], q_c_2[1])

        # qk
        hpk[0] = [x * h for x in pk[0]]
        hpk[1] = [x * h for x in pk[1]]

        qk[0] = add_fft(q_hmc[0], hpk[0])
        qk[1] = add_fft(q_hmc[1], hpk[1])

        # pk
        qk_c[0] = sub_fft(qk[0], t_fft[0])
        qk_c[1] = sub_fft(qk[1], t_fft[1])

        qk_c_2[0] = [x * step for x in qk_c[0]]
        qk_c_2[1] = [x * step for x in qk_c[1]]

        pk[0] = sub_fft(pk[0], qk_c_2[0])
        pk[1] = sub_fft(pk[1], qk_c_2[1])

        return qk, pk

    def calc_v(self, z_fft):

        [[a, b], [c, d]] = self.B0_fft

        v0_fft = add_fft(mul_fft(z_fft[0], a), mul_fft(z_fft[1], c))
        v1_fft = add_fft(mul_fft(z_fft[0], b), mul_fft(z_fft[1], d))
        v0 = [int(round(elt)) for elt in ifft(v0_fft)]
        v1 = [int(round(elt)) for elt in ifft(v1_fft)]

        return v0, v1

    @staticmethod
    def calc_norm(s):
        norm_sign = sum(coef ** 2 for coef in s[0])
        norm_sign += sum(coef ** 2 for coef in s[1])

        return norm_sign

    def sign(self, message, type_in='', sigma_og=None, h=220, k_hmc=3, sigma_new=30, i_mix_sym=1000, overwrite=False, randombytes=urandom):
        """
        Sign a message. The message MUST be a byte string or byte array.
        Optionally, one can select the source of (pseudo-)randomness used
        (default: urandom).
        """
        #start = timer()
        int_header = 0x30 + logn[self.n]
        header = int_header.to_bytes(1, "little")

        salt = randombytes(SALT_LEN)
        hashed = self.hash_to_point(message, salt)

        # We repeat the signing procedure until we find a signature that is
        # short enough (both the Euclidean norm and the bytelength)

        '''Set the original sigma to sample'''
        if sigma_og is not None:
            self.sigma = float(sigma_og)
            self.sigma_old = float(sigma_og)
            self.signature_bound = (1.1 ** 2) * 2 * self.n * (self.sigma ** 2)
            self.T_fft = deepcopy(self.orig_T_fft)
            normalize_tree(self.T_fft, self.sigma)

        while(1):
            if (randombytes == urandom):
                s = self.sample_preimage(hashed, type_in, sigma_new, i_mix_sym, overwrite, h, k_hmc)

            else:
                seed = randombytes(SEED_LEN)
                s = self.sample_preimage(hashed, type_in, sigma_new, i_mix_sym, overwrite, h, k_hmc, seed=seed)
            norm_sign = self.calc_norm(s)
            # Check the Euclidean norm
            if norm_sign <= self.signature_bound:
                enc_s = compress(s[1], self.sig_bytelen - HEAD_LEN - SALT_LEN)
                # Check that the encoding is valid (sometimes it fails)
                if (enc_s is not False):
                    '''
                    Restore T_fft
                    '''
                    self.sigma = Params[self.n]["sigma"]
                    self.signature_bound = floor(Params[self.n]["sig_bound"])
                    self.T_fft = deepcopy(self.orig_T_fft)
                    normalize_tree(self.T_fft, self.sigma)
                    #end = timer()
                    #print("Time elapsed for sign (inside falcon.py): ", end-start, "\n")

                    return header + salt + enc_s

    def verify(self, message, signature):
        """
        Verify a signature.
        """
        # Unpack the salt and the short polynomial s1
        salt = signature[HEAD_LEN:HEAD_LEN + SALT_LEN]
        enc_s = signature[HEAD_LEN + SALT_LEN:]
        s1 = decompress(enc_s, self.sig_bytelen - HEAD_LEN - SALT_LEN, self.n)

        # Check that the encoding is valid
        if (s1 is False):
            print("Invalid encoding")
            return False

        # Compute s0 and normalize its coefficients in (-q/2, q/2]
        hashed = self.hash_to_point(message, salt)
        s0 = sub_zq(hashed, mul_zq(s1, self.h))
        s0 = [(coef + (q >> 1)) % q - (q >> 1) for coef in s0]

        # Check that the (s0, s1) is short
        norm_sign = sum(coef ** 2 for coef in s0)
        norm_sign += sum(coef ** 2 for coef in s1)
        if norm_sign > self.signature_bound:
            print("Squared norm of signature is too large:", norm_sign)
            return False

        # If all checks are passed, accept
        return True
