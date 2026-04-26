# PathogenHunter — Data Pipeline (CORRECT LABELS)

import json
import time
import requests
import os
import random

ORGANISMS = {
    "ecoli": {
        "display_name": "Escherichia coli K-12",
        "deg_organism_id": "107",
        "uniprot_taxon": "83333",
    },
    "saureus": {
        "display_name": "Staphylococcus aureus MRSA252",
        "deg_organism_id": "158",
        "uniprot_taxon": "282458",
    },
    "mtb": {
        "display_name": "Mycobacterium tuberculosis H37Rv",
        "deg_organism_id": "83",
        "uniprot_taxon": "83332",
    },
}

HUMAN_TAXON = "9606"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── helpers ─────────────────────────────────────────────────────────────────

def get_with_retry(url, params=None, retries=3, delay=1.5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


# ── data sources ─────────────────────────────────────────────────────────

def fetch_essential_genes_deg(organism_id: str):
    url = "http://tubic.org/deg/public/online/api/search"
    params = {"organism": organism_id, "format": "json", "limit": 500}

    try:
        r = get_with_retry(url, params=params)
        data = r.json()

        genes = []
        for entry in data.get("data", []):
            genes.append({
                "gene_id": entry.get("locusTag", "unknown"),
                "gene_name": entry.get("geneName", "unknown"),
                "function": entry.get("function", "unknown"),
                "pathway": entry.get("pathway", "unknown"),
                "essential": True,
            })

        return genes

    except Exception:
        print("[DEG failed → using fallback]")
        return _fallback_essential_genes(organism_id)


def check_human_homolog(gene_name: str):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene:{gene_name} AND taxonomy_id:{HUMAN_TAXON}",
        "format": "json",
        "size": 1,
    }

    try:
        r = get_with_retry(url, params=params)
        return len(r.json().get("results", [])) > 0
    except Exception:
        return False


def fetch_binding_compounds(gene_name: str):
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
        r = get_with_retry(url, {"q": gene_name})
        targets = r.json().get("targets", [])

        if not targets:
            return []

        chembl_id = targets[0]["target_chembl_id"]

        r = get_with_retry(
            "https://www.ebi.ac.uk/chembl/api/data/activity.json",
            {"target_chembl_id": chembl_id, "limit": 3}
        )

        return r.json().get("activities", [])

    except Exception:
        return []

def _fallback_essential_genes(organism_id: str) -> list[dict]:
    """
    Curated essential genes per organism with known drug targets highlighted.
    Sourced from: Gerdes et al. 2003, Liu et al. 2021, and DEG v16.
    """
    ecoli_genes = [
        {"gene_id": "b0237", "gene_name": "acpP",  "function": "Acyl carrier protein",                   "pathway": "Fatty acid synthesis", "essential": True},
        {"gene_id": "b0474", "gene_name": "fabI",  "function": "Enoyl-ACP reductase",                    "pathway": "Fatty acid synthesis", "essential": True},
        {"gene_id": "b0553", "gene_name": "murA",  "function": "UDP-GlcNAc enolpyruvyl transferase",     "pathway": "Peptidoglycan synthesis","essential": True},
        {"gene_id": "b0631", "gene_name": "murB",  "function": "UDP-GlcNAc-enolpyruvate reductase",      "pathway": "Peptidoglycan synthesis","essential": True},
        {"gene_id": "b0718", "gene_name": "murC",  "function": "UDP-N-acetylmuramate-alanine ligase",    "pathway": "Peptidoglycan synthesis","essential": True},
        {"gene_id": "b0722", "gene_name": "murD",  "function": "UDP-MurNAc-L-Ala-D-Glu ligase",         "pathway": "Peptidoglycan synthesis","essential": True},
        {"gene_id": "b0739", "gene_name": "murE",  "function": "UDP-MurNAc-L-Ala-D-Glu-meso-DAP ligase","pathway": "Peptidoglycan synthesis","essential": True},
        {"gene_id": "b0170", "gene_name": "dnaA",  "function": "Chromosomal replication initiator",      "pathway": "DNA replication",      "essential": True},
        {"gene_id": "b0184", "gene_name": "gyrB",  "function": "DNA gyrase subunit B",                   "pathway": "DNA replication",      "essential": True},
        {"gene_id": "b0186", "gene_name": "gyrA",  "function": "DNA gyrase subunit A",                   "pathway": "DNA replication",      "essential": True},
        {"gene_id": "b3699", "gene_name": "rpoB",  "function": "DNA-directed RNA polymerase beta chain", "pathway": "Transcription",        "essential": True},
        {"gene_id": "b3987", "gene_name": "rpsA",  "function": "30S ribosomal protein S1",               "pathway": "Translation",          "essential": True},
        {"gene_id": "b0022", "gene_name": "accD",  "function": "Acetyl-CoA carboxylase beta subunit",    "pathway": "Fatty acid synthesis",  "essential": True},
        {"gene_id": "b0025", "gene_name": "lpxC",  "function": "UDP-3-O-acyl GlcNAc deacetylase",        "pathway": "Lipopolysaccharide",    "essential": True},
        {"gene_id": "b0101", "gene_name": "coaA",  "function": "Pantothenate kinase",                    "pathway": "CoA biosynthesis",      "essential": True},
        # Non-essential decoys (agent must learn to avoid these)
        {"gene_id": "b0015", "gene_name": "dnaJ",  "function": "Chaperone protein",                      "pathway": "Stress response",       "essential": False},
        {"gene_id": "b0032", "gene_name": "lacZ",  "function": "Beta-galactosidase",                     "pathway": "Lactose metabolism",    "essential": False},
        {"gene_id": "b0062", "gene_name": "araA",  "function": "L-arabinose isomerase",                  "pathway": "Arabinose metabolism",  "essential": False},
        {"gene_id": "b0098", "gene_name": "thiH",  "function": "Thiamine biosynthesis",                  "pathway": "Thiamine synthesis",    "essential": False},
        {"gene_id": "b0116", "gene_name": "pyrH",  "function": "UMP kinase",                             "pathway": "Pyrimidine biosynthesis","essential": False},
    ]
 
    saureus_genes = [
        {"gene_id": "SA0001", "gene_name": "dnaA",  "function": "Chromosomal replication initiator",   "pathway": "DNA replication",       "essential": True},
        {"gene_id": "SA0006", "gene_name": "gyrB",  "function": "DNA gyrase subunit B",                "pathway": "DNA replication",       "essential": True},
        {"gene_id": "SA0107", "gene_name": "fabI",  "function": "Enoyl-ACP reductase (FabI)",          "pathway": "Fatty acid synthesis",  "essential": True},
        {"gene_id": "SA0234", "gene_name": "murA",  "function": "MurA transferase",                    "pathway": "Cell wall synthesis",   "essential": True},
        {"gene_id": "SA0441", "gene_name": "rpoB",  "function": "RNA polymerase beta subunit",         "pathway": "Transcription",         "essential": True},
        {"gene_id": "SA1146", "gene_name": "ftsZ",  "function": "Cell division protein FtsZ",          "pathway": "Cell division",         "essential": True},
        {"gene_id": "SA1862", "gene_name": "walR",  "function": "Two-component response regulator",    "pathway": "Cell wall homeostasis", "essential": True},
        {"gene_id": "SA0587", "gene_name": "accA",  "function": "Acetyl-CoA carboxylase alpha",        "pathway": "Fatty acid synthesis",  "essential": True},
        {"gene_id": "SA0102", "gene_name": "pheS",  "function": "Phenylalanyl-tRNA synthetase alpha",  "pathway": "tRNA charging",         "essential": True},
        {"gene_id": "SA2094", "gene_name": "mprF",  "function": "Lysyl-phosphatidylglycerol synthase", "pathway": "Membrane modification", "essential": True},
        # Non-essential decoys
        {"gene_id": "SA0073", "gene_name": "agr",   "function": "Accessory gene regulator",            "pathway": "Virulence regulation",  "essential": False},
        {"gene_id": "SA0269", "gene_name": "spa",   "function": "Staphylococcal protein A",             "pathway": "Immune evasion",        "essential": False},
        {"gene_id": "SA1018", "gene_name": "ica",   "function": "Biofilm-associated protein",           "pathway": "Biofilm formation",     "essential": False},
        {"gene_id": "SA2304", "gene_name": "hlb",   "function": "Beta-haemolysin",                      "pathway": "Toxin production",      "essential": False},
    ]
 
    mtb_genes = [
        {"gene_id": "Rv0001", "gene_name": "dnaA",  "function": "Chromosomal replication initiator",    "pathway": "DNA replication",       "essential": True},
        {"gene_id": "Rv0006", "gene_name": "gyrA",  "function": "DNA gyrase subunit A",                 "pathway": "DNA replication",       "essential": True},
        {"gene_id": "Rv1484", "gene_name": "inhA",  "function": "Enoyl-ACP reductase (InhA)",           "pathway": "Mycolic acid synthesis","essential": True},
        {"gene_id": "Rv2245", "gene_name": "kasA",  "function": "Beta-ketoacyl-ACP synthase I",         "pathway": "Fatty acid elongation", "essential": True},
        {"gene_id": "Rv0667", "gene_name": "rpoB",  "function": "RNA polymerase beta subunit",          "pathway": "Transcription",         "essential": True},
        {"gene_id": "Rv3800c","gene_name": "pks13", "function": "Polyketide synthase 13",               "pathway": "Mycolic acid synthesis","essential": True},
        {"gene_id": "Rv2524c","gene_name": "fas",   "function": "Fatty acid synthase",                  "pathway": "Fatty acid synthesis",  "essential": True},
        {"gene_id": "Rv2194", "gene_name": "mmpL3", "function": "Mycolic acid transporter",             "pathway": "Cell wall assembly",    "essential": True},
        {"gene_id": "Rv3283", "gene_name": "folP1", "function": "Dihydropteroate synthase",             "pathway": "Folate synthesis",      "essential": True},
        {"gene_id": "Rv1267c","gene_name": "embR",  "function": "Transcriptional activator of emb",    "pathway": "Arabinan synthesis",    "essential": True},
        # Non-essential decoys
        {"gene_id": "Rv3219", "gene_name": "esxA",  "function": "ESAT-6 secreted antigen",              "pathway": "Secretion system",      "essential": False},
        {"gene_id": "Rv1818c","gene_name": "PE_PGRS","function": "PE-PGRS family protein",              "pathway": "Antigen variation",     "essential": False},
        {"gene_id": "Rv2818c","gene_name": "dosT",  "function": "Oxygen-sensing kinase",                "pathway": "Dormancy response",     "essential": False},
    ]
 
    mapping = {"107": ecoli_genes, "158": saureus_genes, "83": mtb_genes}
    return mapping.get(organism_id, ecoli_genes)
 
 
# known human homologs
# Pre-curated lookup to avoid runtime API calls for common genes
KNOWN_HUMAN_HOMOLOGS = {
    "gyrA": False, "gyrB": False,          # bacterial topoisomerases differ structurally
    "fabI": False,                          # FabI absent in human fatty acid synthesis
    "murA": False, "murB": False,          # peptidoglycan not in humans
    "murC": False, "murD": False, "murE": False,
    "ftsZ": False,                          # bacterial tubulin homolog — structurally distinct
    "inhA": False,                          # mycobacterial-specific InhA
    "rpoB": False,                          # bacterial RNAP beta — structurally distinct from human
    "lpxC": False,                          # LPS pathway absent in humans
    "acpP": True,                           # ACP has human mitochondrial homolog
    "accD": True,                           # ACC has human homolog (ACACB)
    "dnaA": False,                          # distinct from human replication origins
    "coaA": True,                           # pantothenate kinase has human homolog (PANK)
    "rpsA": True,                           # ribosomal proteins have some human homologs
    "kasA": False,                          # KAS enzymes absent in human cytosol
    "mmpL3": False,                         # mycobacterium-specific
    "folP1": False,                         # dihydropteroate synthase absent in humans
    "fas": True,                            # fatty acid synthase — human FASN is a homolog
    "pks13": False,
    "walR": False,
    "mprF": False,
    "accA": True,
    # Non-essential genes
    "dnaJ": True,                           # DnaJ/Hsp40 has human homologs (DNAJA1-4)
    "lacZ": False,                          # no human beta-galactosidase homolog
    "araA": False,                          # no human arabinose isomerase
    "thiH": False,                          # thiamine biosynthesis absent in humans
    "pyrH": True,                           # UMP kinase has human homolog (UMP-CMPK)
    "agr": False,                           # bacterial quorum sensing, no human homolog
    "spa": False,                           # staphylococcal protein A, no human homolog
    "ica": False,                           # biofilm gene, no human homolog
    "hlb": False,                           # beta-haemolysin, no human homolog
    "pheS": True,                           # phenylalanyl-tRNA synthetase has human homolog
    "esxA": False,                          # mycobacterial secretion, no human homolog
    "PE_PGRS": False,                       # mycobacterial-specific
    "dosT": False,                          # mycobacterial dormancy sensor
    "embR": False,                          # mycobacterial transcription factor
}
 
# Known binding compounds per gene (curated from ChEMBL and literature)
KNOWN_BINDING_COMPOUNDS = {
    "fabI":  [{"compound_id": "CHEMBL1346", "compound_name": "Triclosan",     "activity_value": 0.1,   "activity_units": "nM"}],
    "murA":  [{"compound_id": "CHEMBL779",  "compound_name": "Fosfomycin",    "activity_value": 1.2,   "activity_units": "µM"}],
    "inhA":  [{"compound_id": "CHEMBL615",  "compound_name": "Isoniazid",     "activity_value": 0.02,  "activity_units": "µM"},
              {"compound_id": "CHEMBL749",  "compound_name": "Ethionamide",   "activity_value": 1.5,   "activity_units": "µM"}],
    "gyrA":  [{"compound_id": "CHEMBL4",    "compound_name": "Ciprofloxacin", "activity_value": 0.008, "activity_units": "µM"},
              {"compound_id": "CHEMBL715",  "compound_name": "Levofloxacin",  "activity_value": 0.012, "activity_units": "µM"}],
    "gyrB":  [{"compound_id": "CHEMBL487",  "compound_name": "Novobiocin",    "activity_value": 0.1,   "activity_units": "µM"}],
    "rpoB":  [{"compound_id": "CHEMBL374",  "compound_name": "Rifampicin",    "activity_value": 0.015, "activity_units": "µM"}],
    "ftsZ":  [{"compound_id": "CHEMBL4523038","compound_name":"PC190723",     "activity_value": 0.18,  "activity_units": "µM"}],
    "lpxC":  [{"compound_id": "CHEMBL2178717","compound_name":"LpxC-1",       "activity_value": 0.005, "activity_units": "µM"}],
    "mmpL3": [{"compound_id": "CHEMBL4523126","compound_name":"SQ109",        "activity_value": 0.054, "activity_units": "µM"}],
    "folP1": [{"compound_id": "CHEMBL1359", "compound_name": "Sulfamethoxazole","activity_value":1.1,  "activity_units": "µM"}],
    "kasA":  [{"compound_id": "CHEMBL1347", "compound_name": "Thiolactomycin","activity_value": 2.3,  "activity_units": "µM"}],
    "coaA":  [{"compound_id": "CHEMBL606",  "compound_name": "Pantetheine",   "activity_value": 45.0, "activity_units": "µM"}],
}


# ── main build ───────────────────────────────────────────────────────────────

def build_gene_database(organism_key):
    org = ORGANISMS[organism_key]

    print(f"\nBuilding {org['display_name']}")

    genes = fetch_essential_genes_deg(org["deg_organism_id"])

    enriched = []

    for g in genes:
        name = g["gene_name"]

        # Use curated lookup; fall back to API only if unknown
        has_h = KNOWN_HUMAN_HOMOLOGS.get(name)
        if has_h is None:
            has_h = check_human_homolog(name)
            time.sleep(0.2)

        # Use curated binding data; fall back to API only if unknown
        binding = KNOWN_BINDING_COMPOUNDS.get(name)
        if binding is None:
            binding = fetch_binding_compounds(name)
            time.sleep(0.2)

        # Compute valid target flag using real biological labels
        is_valid = (
            g["essential"]
            and not has_h
            and len(binding) > 0
        )

        enriched.append({
            **g,
            "has_human_homolog": has_h,
            "binding_compounds": binding,
            "is_valid_target": is_valid,
        })

    valid_count = sum(g["is_valid_target"] for g in enriched)
    print(f"Valid targets: {valid_count} / {len(enriched)}")

    output = {
        "organism_key": organism_key,
        "display_name": org["display_name"],
        "genes": enriched,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"genes_{organism_key}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved → {path}")


# ── run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ORGANISMS:
        build_gene_database(key)