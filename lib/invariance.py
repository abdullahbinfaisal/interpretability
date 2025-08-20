import torch



class InvarianceStudy:
    def __init__(self, sparse_mlps):
        self.sparse_mlps = {}
        
        for key in sparse_mlps.keys():
            self.sparse_mlps[key] = sparse_mlps[key].weight.detach().cpu()
            self.num_classes = self.sparse_mlps[key].shape[0]

    def aggregator(self, normalize=True):
        raw_matrices = [w for _, w in self.sparse_mlps.items()]
        normalized_matrices = []

        if normalize:
            for w in raw_matrices:
                max_abs = w.abs().max()
                w_norm = w / max_abs
                normalized_matrices.append(w_norm)

        aggregated = torch.abs(torch.stack(normalized_matrices if normalize else raw_matrices, dim=0)).sum(dim=0)
        
        return aggregated


    def get_domain_concepts(self, cls, domain):
        
        weights = self.sparse_mlps[domain][cls]   # 1D tensor of weights (can be +/- or 0)
        return self.extract_concepts(weights)


    def get_common_concepts(self, cls, normalize=True):
        weights = self.aggregator(normalize=normalize)[cls]
        return self.extract_concepts(weights)




    def extract_concepts(self, weights):
        nonzero_mask = weights != 0
        nonzero_idx = torch.nonzero(nonzero_mask, as_tuple=True)[0]

        pos_mask = weights > 0
        neg_mask = weights < 0

        pos_idx = torch.nonzero(pos_mask, as_tuple=True)[0]
        neg_idx = torch.nonzero(neg_mask, as_tuple=True)[0]

        pos_vals, pos_sort_idx = torch.sort(weights[pos_idx], descending=True)
        neg_vals, neg_sort_idx = torch.sort(weights[neg_idx], descending=False)  # more negative first

        pos_features = pos_idx[pos_sort_idx]
        neg_features = neg_idx[neg_sort_idx]

        pos_list = [{"concept": int(f.item()), "weight": float(v.item())} for f, v in zip(pos_features, pos_vals)]
        neg_list = [{"concept": int(f.item()), "weight": float(v.item())} for f, v in zip(neg_features, neg_vals)]

        return {
            "positive" : pos_list,
            "negative" : neg_list
        }
    
    def print_domain_concepts(self, cls, domain):
        print(f"\nClass {cls} | Domain: {domain}")
        concepts = self.get_domain_concepts(cls, domain)

        for c in concepts["positive"]:
            print(f"    Concept {c['concept']} -> \033[92m{c['weight']:.5f}\033[0m")

        for c in concepts["negative"]:
            print(f"    Concept {c['concept']} -> \033[91m{c['weight']:.5f}\033[0m")
        return

    def print_common_concepts(self, cls, normalize=True):
        print(f"\nClass {cls} | Aggregated Concepts (normalize={normalize})")
        agg_concepts = self.get_common_concepts(cls, normalize=normalize)

        for c in agg_concepts["positive"]:
            print(f"    Concept {c['concept']} -> \033[92m{c['weight']:.5f}\033[0m")

        for c in agg_concepts["negative"]:
            print(f"    Concept {c['concept']} -> \033[91m{c['weight']:.5f}\033[0m")
        return

    def print_all(self):
        for cls in range(self.num_classes):
            print(f"\nClass {cls} ============================================================")

            # Per-domain concepts
            for domain, weights in self.sparse_mlps.items():
                print(f"\033[94mDomain: {domain}\033[0m")
                concepts = self.get_domain_concepts(cls, domain)

                for c in concepts["positive"]:
                    print(f"    Concept {c['concept']} -> \033[92m{c['weight']:.5f}\033[0m")

                for c in concepts["negative"]:
                    print(f"    Concept {c['concept']} -> \033[91m{c['weight']:.5f}\033[0m")

            # Aggregated concepts
            print(f"\033[94mAggregated\033[0m")
            agg_concepts = self.get_common_concepts(cls)
            for c in agg_concepts["positive"]:
                print(f"    Concept {c['concept']} -> \033[92m{c['weight']:.5f}\033[0m")

            for c in agg_concepts["negative"]:
                print(f"    Concept {c['concept']} -> \033[91m{c['weight']:.5f}\033[0m")                
        return
