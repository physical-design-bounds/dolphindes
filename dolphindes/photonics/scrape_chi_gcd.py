import numpy as np
import scipy.sparse as sp
import sys

sys.path.append("../../../")

import dualbound.Maxwell.TM_FDFD as TM
import Examples.verlan_gcd.inverse_design.design_tools_nojax as dt
import json, os
import copy
import matplotlib.pyplot as plt
import time
from dualbound.Constraints.off_diagonal_P_save_struct import (
    dual_space_reduction_iteration_Msparse_align_mineig_maxviol as gcd,
)

# Implements scraping for naive_extraction_gcd
# Basically just take yn -> yn + sigma * (previous optimal T) until you get strong duality
# Strong duality will be checked by the condition that the compact constraint holds

theta = float(sys.argv[1])
gcd_max_iter = eval(str(sys.argv[4]))
start_viter = int(sys.argv[5])
delta_t = float(sys.argv[8])
n_theta = int(sys.argv[9])
delta_t_reduct = float(sys.argv[10])


def vn(value):  # vn = value_name
    value_str = f"{value}"
    return value_str.replace(".", "p")


theta = np.radians(theta)
verlan_params = {}

zinv = np.imag(chi) / np.real(chi * np.conj(chi))

Evac = TM.get_TM_dipole_field(omega, dl, Nx, Ny, cx, cy, Npml)
vacLDOS = -0.5 * np.real(Evac[cx, cy])
print("vacLDOS", vacLDOS, flush=True)
Si_desvec = Evac[des_mask]
O_lin = (-1j * omega / 4) * (Ginv.conj().T @ Si_desvec.conj()) * dl**2
O_lin_dense = (-1j * omega / 4) * (Si_desvec.conj()) * dl**2
O_lin_dense_norm = np.linalg.norm(O_lin_dense)

O_quad = sp.csc_matrix(Ginv.shape, dtype=complex)
dualconst = vacLDOS

verlan_result = None


def dual_func(yn, verlan_iter, plot_struct, gcd_chi, only_viol=False):
    viter_folder = f"{folder}/viter={verlan_iter}/"
    if isinstance(verlan_iter, int):
        prev_viter_folder = f"{folder}/viter={verlan_iter - 1}/"

    savefunc = lambda optstruct, iternum, mindual, lags=None: structFunc(
        optstruct, iternum, mindual, lags, verlan_iter, plot_struct
    )

    if verlan_iter == 0 or not isinstance(verlan_iter, int):
        prev_Pdatalist = None
        prev_optLags = None
    else:
        prev_Pdatalist = np.load(prev_viter_folder + f"final_struct_Pdatalist.npy")
        prev_optLags = np.load(prev_viter_folder + f"final_struct_optLags.npy")

    Pdatalist, optLags, mindual, mingrad, optGT, time_data = gcd(
        gcd_chi,
        Si_desvec,
        Ginv,
        yn,
        O_quad,
        dualconst=dualconst,
        structFunc=savefunc,
        outputFunc=None,
        Pnum=pnum,
        save_period=5,
        gcd_max_iter=gcd_max_iter,
        gcd_iter_period=gcd_iter_period,
        verbose=0,
        Plist_start=prev_Pdatalist,
        Lags_start=prev_optLags,
        only_viol=only_viol,
    )
    Pdatalist = np.array(Pdatalist)

    optT = Ginv @ optGT
    result = {
        "dualval": mindual,
        "compact_viol": mingrad[1],
        "all_viol": mingrad,
        "GT": optGT,
        "T": optT,
    }

    viter_folder = f"{folder}/viter={verlan_iter}/"
    np.save(viter_folder + f"final_struct_Pdatalist.npy", Pdatalist)
    np.save(viter_folder + f"final_struct_optLags.npy", optLags)
    np.save(viter_folder + f"final_struct_viol.npy", mingrad)
    np.save(viter_folder + f"final_struct_T.npy", optT)
    np.save(viter_folder + f"gcd_time_data.npy", np.array(time_data))

    return result


def scrape(yn_dense, result, param, mode):
    # param is sigma or theta depending on mode
    assert mode in ["sigma", "theta"]
    T = result["T"] * O_lin_dense_norm / np.linalg.norm(result["T"])

    if mode == "sigma":
        yn_dense = yn_dense + param * T
    elif mode == "theta":
        original_yn_dense = yn_dense.copy()
        while True:
            yn_dense = yn_dense + T
            yn_dense *= O_lin_dense_norm / np.linalg.norm(yn_dense)
            angle = np.arccos(
                np.clip(
                    np.dot(original_yn_dense.conj().T, yn_dense)
                    / (np.linalg.norm(original_yn_dense) * np.linalg.norm(yn_dense)),
                    -1.0,
                    1.0,
                )
            )
            if np.allclose(yn_dense, T, 1e-3):
                too_close = True
                break
            elif angle >= param:
                too_close = False
                break

    yn_dense *= O_lin_dense_norm / np.linalg.norm(yn_dense)
    yn = Ginv.conj().T @ yn_dense
    return yn, yn_dense, too_close


def chi_scrape(oyn_dense, delta_t, theta, t_start=1, start_viter=0):
    if start_viter == 0:
        yn_dense = oyn_dense
    else:
        yn_dense = np.load(f"{folder}/viter={start_viter - 1}/yn_dense.npy")
    yn = Ginv.conj().T @ yn_dense

    verlan_iter = start_viter
    plot_struct = True
    only_viol = False
    original_delta_t = np.copy(delta_t)

    # First, inflate with the original objective
    dfunc = lambda t, name: dual_func(
        yn, name, plot_struct, chi_t(t), only_viol=only_viol
    )

    print()
    print("Deflation.")
    print("----------")
    t_SD, result, success = binary_search_SD(0, t_start, dfunc, tol=0.5)
    assert success, "Deflation failed."

    if t_SD == 1:
        print("Original problem strongly dual. Done.")
        return

    print(f"Success: chi_SD0 = {chi_t(t_SD)}")
    print()

    failure_counter = 0
    success_counter = 0

    if viol_switch:
        only_viol = True

    print("Inflation.")
    print("----------")

    while True:
        viter_folder = f"{folder}/viter={verlan_iter}/"
        os.makedirs(viter_folder, exist_ok=True)
        if start_viter != verlan_iter or verlan_iter == 0 and success:
            np.save(viter_folder + f"yn.npy", yn)
            np.save(viter_folder + f"yn_dense.npy", yn_dense)
            np.save(
                viter_folder + f"yn_overlap.npy",
                np.abs(oyn_dense.conj() @ yn_dense)
                / np.linalg.norm(oyn_dense)
                / np.linalg.norm(yn_dense),
            )
            np.save(viter_folder + f"chi.npy", chi_t(t_SD))

        if success:
            failure_counter = 0
            success_counter += 1

            if success_counter == 2:
                delta_t = original_delta_t

            print(
                f"Succeeded with t_SD = {t_SD}. delta_t = {delta_t}, t_new = {np.clip(t_SD + delta_t, 0, 1)}."
            )
            print("---------------------------------------")

            if t_SD + delta_t >= 1:
                delta_t = 1 - t_SD
            t_new = np.clip(t_SD + delta_t, 0, 1)

            verlan_iter += 1
        else:
            success_counter = 0
            failure_counter += 1
            if (delta_t_reduct ** (1 + failure_counter)) < 1e-5:
                print("Reducing t by less than 1e-5, stopping. Failed.")
                break

            delta_t = delta_t * (delta_t_reduct ** (1 + failure_counter))

            print(
                f"Succeeded with t_SD = {t_SD}. delta_t = {delta_t}, t_new = {np.clip(t_SD + delta_t, 0, 1)}."
            )
            print("---------------------------------------")

            t_new = t_SD + delta_t

        if np.isclose(t_SD, 1):
            print("Found strong duality! Done.")
            break

        scrape_counter = 0
        y_dense_new, result_new = yn_dense, result  # Starting yn

        print(
            f"Scraping at iteration={verlan_iter}.", flush=True
        ) if success else print(
            f"Re-trying scraping at iteration={verlan_iter}.", flush=True
        )
        while True:
            scrape_counter += 1

            yn_new, yn_dense_new, too_close = scrape(
                y_dense_new, result_new, theta, "theta"
            )

            print("Scraping made yn = T.") if too_close else print(
                f"Scraped to angle {np.degrees(theta)}."
            )
            print(
                f"Overlap is {np.abs(oyn_dense.conj() @ yn_dense_new) / np.linalg.norm(oyn_dense) / np.linalg.norm(yn_dense_new)}"
            )

            dfunc = lambda t, _=None: dual_func(
                yn_new, verlan_iter, plot_struct, chi_t(t), only_viol
            )

            t1 = time.time()
            result_new = dfunc(np.clip(t_new, 0, 1), None)
            t2 = time.time()

            if _check_strong_duality(result_new["compact_viol"], delta):
                print(
                    f"Strong duality found with n_scape = {scrape_counter}. time = {t2 - t1}"
                )
                success = True
                break
            else:
                print(
                    f"Failed to find strong duality step. Scraping and trying again... n_scrape = {scrape_counter}. time = {t2 - t1}"
                )
                success = False

            if scrape_counter >= n_theta:
                print(
                    f"Scraped {scrape_counter} >= n_theta times. Failed inner loop, reduce delta_t!"
                )
                failure_counter += 1
                break

        if success:
            yn_dense = yn_dense_new
            yn = yn_new
            result = result_new
            t_SD = t_new


chi_scrape(O_lin_dense, delta_t, theta, t_start=1, start_viter=0)
