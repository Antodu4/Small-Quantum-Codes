#!/usr/bin/env python3

import requests
import re
import sys


FORMAT_MAP = {
    "gaussian": "gaussian94",
    "orca": "orca",
    "qchem": "qchem",
    "psi4": "psi4",
    "gamess": "gamess_us",
}


# ==============================
# TELECHARGEMENT BSE
# ==============================

def download_basis(basis_name, element, code):

    if code not in FORMAT_MAP:
        raise ValueError("Code non supporté.")

    fmt = FORMAT_MAP[code]

    url = f"https://www.basissetexchange.org/api/basis/{basis_name}/format/{fmt}?elements={element}"
    r = requests.get(url)

    if r.status_code != 200:
        raise Exception("Téléchargement échoué.")

    return r.text


# ==============================
# EXTRACTION EXPOSANTS
# ==============================

def extract_min_exponents(text):

    exponents = {}
    current_orb = None

    for line in text.split("\n"):

        header = re.match(r"^\s*([SPDFGH])\s+\d+", line)
        if header:
            current_orb = header.group(1)
            if current_orb not in exponents:
                exponents[current_orb] = []
            continue

        value = re.match(r"^\s*([0-9\.\-DE\+]+)\s+", line)
        if value and current_orb:
            try:
                val = float(value.group(1).replace("D", "E"))
                exponents[current_orb].append(val)
            except:
                pass

    return {orb: min(vals) for orb, vals in exponents.items() if vals}


# ==============================
# GENERATION SERIE
# ==============================

def generate_series(alpha0, k, n):
    return [alpha0 / (k ** i) for i in range(n)]


# ==============================
# CREATION EX-NIHILO
# ==============================

def create_ex_nihilo(element, code, extensions):

    lines = []

    if code == 'orca':
        lines.append(f"# Ex-nihilo basis for {element}")
        for orb, params in extensions.items():
            series = generate_series(params["alpha0"], params["k"], params["n"])
            for exp in series:
                lines.append(f"{orb}   1    ")
                lines.append(f"1         {exp:.10E}           1.000000")

    elif code == 'gamess':
        lines.append("")
        lines.append("$DATA")
        lines.append(element)
        for orb, params in extensions.items():
            series = generate_series(params["alpha0"], params["k"], params["n"])
            for exp in series:
                lines.append(f"{orb}   1    ")
                lines.append(f"1         {exp:.10E}           1.000000")
        lines.append("$END")

    else:  # gaussian94, qchem, psi4
        lines.append("$basis")
        lines.append(f"{element}     0")
        for orb, params in extensions.items():
            series = generate_series(params["alpha0"], params["k"], params["n"])
            for exp in series:
                lines.append(f"{orb}    1   1.00")
                formatted_exp = f"{exp:.10E}".replace("E", "D")
                lines.append(f"      {formatted_exp}           1.000000")
        lines.append("****")
        lines.append("$end")
        lines.append("")
    return "\n".join(lines)


# ==============================
# INSERTION APRES DERNIER BLOC
# ==============================

def inject_primitives(text, extensions, code):

    min_exp = extract_min_exponents(text)

    # Ajustement intelligent
    for orb in extensions:
        if orb in min_exp:
            alpha_min = min_exp[orb]
            alpha0 = extensions[orb]["alpha0"]
            k = extensions[orb]["k"]

            if alpha0 >= alpha_min:
                new_alpha0 = alpha_min / k
                print(f"\n[INFO] {orb} : alpha0 ajusté → {new_alpha0:.6E}")
                extensions[orb]["alpha0"] = new_alpha0

    lines = text.split("\n")

    # Trouver la dernière ligne de chaque bloc orbital
    last_line_of_block = {}
    current_orb = None
    last_orb_line = {}
    
    for i, line in enumerate(lines):
        header = re.match(r"^\s*([SPDFGH])\s+\d+", line)
        if header:
            orb = header.group(1)
            current_orb = orb
            last_orb_line[orb] = i
        elif current_orb and re.match(r"^\s*([0-9\.\-DE\+]+)\s+", line):
            # C'est une ligne d'exposant, on met à jour la dernière ligne de ce bloc
            last_orb_line[current_orb] = i

    # Ne garder que les orbitales qui nous intéressent
    for orb in extensions:
        if orb in last_orb_line:
            last_line_of_block[orb] = last_orb_line[orb]

    # Construire nouvelles lignes
    new_lines = []
    i = 0

    while i < len(lines):
        new_lines.append(lines[i])

        # Si cette ligne est la dernière ligne d'un bloc orbital à étendre
        for orb in extensions:
            if orb in last_line_of_block and i == last_line_of_block[orb]:

                alpha0 = extensions[orb]["alpha0"]
                k = extensions[orb]["k"]
                n = extensions[orb]["n"]

                series = generate_series(alpha0, k, n)

                if code == 'orca':
                    for exp in series:
                        new_lines.append(f"{orb}   1    ")
                        new_lines.append(f"1         {exp:.10E}           1.000000")
                elif code == 'gamess':
                    for exp in series:
                        new_lines.append(f"{orb}   1    ")
                        new_lines.append(f"1         {exp:.10E}           1.000000")
                else:
                    for exp in series:
                        new_lines.append(f"{orb}    1   1.00")
                        formatted_exp = f"{exp:.10E}".replace("E", "D")
                        new_lines.append(f"      {formatted_exp}           1.000000")
        i += 1

    # Ajouter les blocs d'orbitales absentes du fichier source
    for orb in extensions:
        if orb not in last_line_of_block:
            alpha0 = extensions[orb]["alpha0"]
            k = extensions[orb]["k"]
            n = extensions[orb]["n"]
            series = generate_series(alpha0, k, n)
            print(f"\n[INFO] {orb} : bloc absent, ajout en fin de fichier ({n} primitives, alpha0={alpha0:.6E}, k={k})")
            if code in ('orca', 'gamess'):
                for exp in series:
                    new_lines.append(f"{orb}   1    ")
                    new_lines.append(f"1         {exp:.10E}           1.000000")
            else:
                for exp in series:
                    new_lines.append(f"{orb}    1   1.00")
                    formatted_exp = f"{exp:.10E}".replace("E", "D")
                    new_lines.append(f"      {formatted_exp}           1.000000")

    return "\n".join(new_lines)


# ==============================
# MAIN
# ==============================

def main():

    mode = input("Mode (étendre/créer) : ").lower().strip()
    element = input("Élément (ex: C) : ")
    code = input("Code (gaussian/orca/qchem/psi4/gamess) : ").lower()

    if mode == "étendre":
        basis_name = input("Nom de la base : ")
        try:
            basis_text = download_basis(basis_name, element, code)
            print("Base téléchargée.")
        except:
            file_path = input("Téléchargement échoué. Fichier local : ")
            with open(file_path, "r") as f:
                basis_text = f.read()

    print("\n--- Paramétrage des orbitales ---")

    prompt = "Orbitales à étendre" if mode == "étendre" else "Orbitales à créer"
    orbitals = input(f"{prompt} (ex: S,P,D) : ").upper()

    if orbitals.strip() == "":
        print("Aucune orbitale spécifiée.")
        sys.exit()

    orb_list = [o.strip() for o in orbitals.split(",")]

    n_list = input(
        f"Nombre de gaussiennes pour {','.join(orb_list)} (ex: 10,5,5) : "
    ).split(",")

    alpha_list = input(
        f"Alpha initiaux pour {','.join(orb_list)} (ex: 1e-2,5e-3,5e-3) : "
    ).split(",")

    k_list = input(
        f"Facteurs k pour {','.join(orb_list)} (ex: 2,2,3) : "
    ).split(",")

    if not (len(orb_list) == len(n_list) == len(alpha_list) == len(k_list)):
        raise ValueError("Les listes fournies n'ont pas la même longueur.")

    extensions = {}
    for orb, n, alpha0, k in zip(orb_list, n_list, alpha_list, k_list):
        extensions[orb] = {
            "n": int(n),
            "alpha0": float(alpha0),
            "k": float(k)
        }

    if mode == "étendre":
        result = inject_primitives(basis_text, extensions, code)
        output_file = f"{basis_name}_{element}_{code}_extended.txt"
    else:
        result = create_ex_nihilo(element, code, extensions)
        output_file = f"exnihilo_{element}_{code}.txt"

    with open(output_file, "w") as f:
        f.write(result)

    print(f"\nBase écrite dans {output_file}")


if __name__ == "__main__":
    main()

