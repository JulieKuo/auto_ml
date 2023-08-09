import os

def save_explain_plots(model, data, type_, chart_root):
    # print("get explain objet")
    obj = model.explain(data, exclude_explanations = ["ice", "pdp"], render=False)

    path = os.path.join(chart_root, "chart", type_)
    os.makedirs(path, exist_ok=True)

    for key in obj.keys():
        # print(f"saving {key} plots")
        if (not obj.get(key).get("plots")):
            continue

        plots = obj.get(key).get("plots")
        try:
            fig = plots.figure()
            fig.savefig(os.path.join(path, f"{key}.png"))

        except:
            plots_key = plots.keys()

            for key1 in plots_key:
                plots1 = plots.get(key1)
                if "H2OExplanation" in str(plots1):
                    plots_key1 = plots1.keys()
                    for key2 in plots_key1:
                        plots2 = plots1.get(key2)
                        fig = plots2.figure()
                        fig.savefig(os.path.join(path, f"{key}.png"))
                
                else:
                    fig = plots1.figure()
                    fig.savefig(os.path.join(path, f"{key}.png"))