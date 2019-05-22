import csv


def hyper_to_hypo_graph(filename, g, gephi_out=None, delimiter='\t'):
    print("Write output to: %s" % filename)

    with open(filename, "w+") as f:
        writer = csv.writer(f, delimiter=delimiter)
        id = 0

        for hyper in g:
            for hypo in g[hyper]:
                row = [id, hypo, hyper]
                writer.writerow(row)
                id += 1

    if gephi_out is not None and gephi_out != "":
        print("Write gephi output to: %s" % gephi_out)

        with open(gephi_out, "w+") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(["id", "source", "target"])
            id = 0

            for hyper in g:
                for hypo in g[hyper]:
                    row = [id, hypo, hyper]
                    writer.writerow(row)
                    id += 1


def network_graph(filename, g, gephi_out=None, delimiter='\t'):
    print("Write output to: %s" % filename)

    with open(filename, "w+") as f:
        writer = csv.writer(f, delimiter=delimiter)
        id = 0

        for hypo in g:
            for hyper in g[hypo]:
                row = [id, hypo, hyper]
                writer.writerow(row)
                id += 1

    if gephi_out is not None and gephi_out != "":
        print("Write gephi output to: %s" % gephi_out)

        with open(gephi_out, "w+") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(["id", "source", "target"])
            id = 0

            for hypo in g:
                for hyper in g[hypo]:
                    row = [id, hypo, hyper]
                    writer.writerow(row)
                    id += 1
