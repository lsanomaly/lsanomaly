"""
**generate_latex.py**

A commandline application to create a latex table summarising the results of
LSAnomaly static data experiments. This is a refactored version of
the script in `evaluate_lsanomaly.zip`
(see https://cit.mak.ac.ug/staff/jquinn/software/lsanomaly.html).

**usage**

generate_latex.py [-h] --input-json JSON_FILE --latex-output LATEX_FILE

Create a LaTeX document with a table of results

**Arguments**

-h, --help
    show this help message and exit

--input-json JSON_FILE, -i JSON_FILE
    path and file name of the results

--latex-output LATEX_FILE, -o LATEX_FILE
    path and file name of the LaTeX file

"""
import json
import logging

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

fmt = "[%(asctime)s %(levelname)-8s] [%(filename)s:%(lineno)4s - %(funcName)s()] %(message)s"  # noqa
logging.basicConfig(level=logging.INFO, format=fmt)


def results_table(json_results):
    """
    Build the LaTeX table.

    Args:
        json_results (dict): results from `run_eval`

    Returns:
        str: LaTeX table

    """
    datasets = sorted(list(json_results["auc"]))
    n_methods = len(json_results["auc"][datasets[0]][0])
    n_methods -= 1

    tex_list = list()

    tex_list.append(r"\begin{tabular}{@{}l%s@{}}" % ("c" * (n_methods + 2)))
    tex_list.append(r"\toprule")
    tex_list.append(
        r"\multicolumn{3}{c}{Dataset} & \multicolumn{%d}{c}{Method} \\"
        % n_methods
    )
    tex_list.append(
        r"\cmidrule(r){1-3}\morecmidrules\cmidrule(l){4-%d}"
        % (4 + n_methods - 1)
    )
    tex_list.append(r"& d & N & LSAD & OCSVM & KNN & KM \\")
    tex_list.append(r"\midrule")

    for dataset in datasets:
        if not None in json_results["auc"][dataset]:  # noqa
            # find out which of the methods is the highest average score
            auc_means = np.zeros(n_methods)
            for method in range(n_methods):
                auc_means[method] = np.mean(
                    json_results["auc"][dataset][method]
                )
            best_methods = [auc_means.argmax()]

            is_best_unique = True
            for method in range(n_methods):
                if method != best_methods[0]:
                    if auc_means[best_methods[0]] < auc_means[method] + 1e-4:
                        is_best_unique = False

            # find out which methods are statistically equivalent in performance  # noqa
            for method in range(n_methods):
                if method != best_methods[0]:
                    p_value = scipy.stats.ttest_rel(
                        json_results["auc"][dataset][best_methods[0]],
                        json_results["auc"][dataset][method],
                    )[1]
                    if p_value > 0.05:
                        best_methods.append(method)

            dataset_name = dataset.replace("_", "-")
            tex_list.append("%s &" % (dataset_name)),
            tex_list.append("%d &" % (json_results["datasize"][dataset][1])),
            tex_list.append("%d &" % (json_results["datasize"][dataset][0])),

            for method in range(n_methods):
                if method == best_methods[0] and is_best_unique:
                    tex_list.append(r"\textbf{%.4f}" % (auc_means[method])),
                elif method in best_methods:
                    tex_list.append(r"\emph{%.4f}" % (auc_means[method])),
                else:
                    tex_list.append("%.4f" % (auc_means[method])),
                if method < (n_methods - 1):
                    tex_list.append("&"),
            tex_list.append(r"\\")

    tex_list.append(r"\midrule")
    tex_list.append(r"Time & & &")

    reference = None

    for method in range(n_methods):
        t = 0.1
        for dataset in datasets:
            if not None in json_results["auc"][dataset]:  # noqa
                t += json_results["time"][dataset][method]
        if not reference:
            reference = t
        t = t / reference
        tex_list.append("%.2f" % t)
        if method < (n_methods - 1):
            tex_list.append("&"),
    tex_list.append(r"\\")

    tex_list.append(r"\bottomrule")
    tex_list.append(r"\end{tabular}")

    tex_out = "\n".join(tex_list)

    return tex_out


def main(input_json, output_latex):
    """
    Read the JSON results file; generate the table in LaTeX;
    wrap the results table in a simple LaTeX document and write it to
    `output_latex`

    Args:
        input_json (str): file of the JSON-serialized results

        output_latex (str): file where the LaTeX document will be written

    """
    here = os.path.abspath(os.path.dirname(__file__))

    try:
        with open(input_json) as fp:
            results_dict = json.load(fp)

        tex_table = results_table(results_dict)

        with open(os.path.join(here, "tex", "front.tex"), "r") as ft:
            front = ft.readlines()

        with open(os.path.join(here, "tex", "back.tex"), "r") as bt:
            back = bt.readlines()

        with open(output_latex, "w") as tex_output:
            tex_output.writelines(front)
            tex_output.write(tex_table)
            tex_output.writelines(back)

        logger.info("LaTeX written to {}".format(output_latex))

    except (FileNotFoundError, json.JSONDecodeError):
        raise


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Create a LaTeX document with a table of results"
    )
    parser.add_argument(
        "--input-json",
        "-i",
        dest="json_file",
        required=True,
        help="path and file name of the results",
    )
    parser.add_argument(
        "--latex-output",
        "-o",
        dest="latex_file",
        required=True,
        help="path and file name of the LaTeX file",
    )
    args = parser.parse_args()
    sys.exit(main(args.json_file, args.latex_file))
