import os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from biotite.structure.io import pdb
from biotite.sequence.io import fasta
from loguru import logger

from rocketshp import RocketSHP, load_sequence, load_structure


class BatchProcessor:
    """
    Efficiently process multiple proteins through RocketSHP.
    
    This class handles batch processing of proteins (from PDB structures or
    sequences) with proper device management and optimized batching.
    """
    
    def __init__(
        self,
        model: RocketSHP,
        batch_size: int = 8,
        device: str = None,
        temperature: float = 300.0,
        use_structure: bool = True
    ):
        """
        Initialize the batch processor.
        
        Args:
            model: A loaded RocketSHP model instance
            batch_size: Maximum number of proteins to process in a batch
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection)
            temperature: Simulation temperature in Kelvin
            use_structure: Whether to use structure information when available
        """
        self.model = model
        self.batch_size = batch_size
        self.use_structure = use_structure
        self.temperature = temperature
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _detect_file_type(self, filename: str) -> str:
        """Detect if the file is a PDB structure or sequence file."""
        ext = Path(filename).suffix.lower()
        if ext in ['.pdb', '.cif', '.mmcif']:
            return 'structure'
        elif ext in ['.fasta', '.fa', '.seq', '.txt']:
            return 'sequence'
        else:
            # Try to infer based on content
            with open(filename, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('>'):
                    return 'sequence'
                elif first_line.startswith('ATOM') or first_line.startswith('HEADER'):
                    return 'structure'
        
        raise ValueError(f"Cannot determine file type for {filename}")
    
    def _load_input(self, filename: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load sequence and optionally structure features from a file."""
        file_type = self._detect_file_type(filename)
        
        if file_type == 'structure':
            # Load structure file
            structure = pdb.PDBFile.read(filename)
            # Extract sequence from structure
            sequence = structure.get_sequence()
            
            seq_feats = load_sequence(sequence)
            
            if self.use_structure:
                struct_feats = load_structure(structure)
            else:
                struct_feats = None
                
        else:  # sequence file
            # Load sequence file
            seq_record = list(fasta.FastaFile.read(filename).items())[0]
            sequence = seq_record[1]
            
            seq_feats = load_sequence(sequence)
            struct_feats = None
            
        return seq_feats, struct_feats
    
    def _create_batch(self, inputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """Create a batch from a list of inputs with padding."""
        # Group similar length sequences for efficiency
        inputs.sort(key=lambda x: x['seq_feats'].shape[0], reverse=True)
        
        # Get max sequence length in this batch
        max_len = inputs[0]['seq_feats'].shape[0]
        
        # Prepare batch tensors
        batch_size = len(inputs)
        seq_feats_batch = torch.zeros(
            batch_size, max_len, inputs[0]['seq_feats'].shape[1], 
            dtype=inputs[0]['seq_feats'].dtype
        )
        
        # Prepare structure features if available
        if self.use_structure and inputs[0]['struct_feats'] is not None:
            struct_dim = inputs[0]['struct_feats'].shape[1]
            struct_feats_batch = torch.zeros(
                batch_size, max_len, struct_dim,
                dtype=inputs[0]['struct_feats'].dtype
            )
        else:
            struct_feats_batch = None
            
        # Prepare sequence masks for actual lengths
        seq_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        # Fill batch tensors
        for i, input_dict in enumerate(inputs):
            seq_len = input_dict['seq_feats'].shape[0]
            seq_feats_batch[i, :seq_len] = input_dict['seq_feats']
            seq_mask[i, :seq_len] = True
            
            if struct_feats_batch is not None and input_dict['struct_feats'] is not None:
                struct_feats_batch[i, :seq_len] = input_dict['struct_feats']
                
        # Prepare temperature tensor
        temp_batch = torch.ones(batch_size, 1) * self.temperature
                
        # Create batch dictionary
        batch = {
            'seq_feats': seq_feats_batch,
            'seq_mask': seq_mask,
            'temp': temp_batch
        }
        
        if struct_feats_batch is not None:
            batch['struct_feats'] = struct_feats_batch
        else:
            batch['struct_feats'] = None
            
        return batch
    
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
        """Process a batch through the model and return unbatched results."""
        # Move batch to device
        device_batch = {
            k: v.to(self.device) if v is not None and isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Run inference
        with torch.no_grad():
            batch_output = self.model(device_batch)
            
        # Unbatch results using sequence mask
        results = []
        seq_mask = batch['seq_mask']
        
        for i in range(seq_mask.shape[0]):
            seq_len = seq_mask[i].sum().item()
            
            # Extract results for this sequence
            result = {
                'rmsf': batch_output['rmsf'][i, :seq_len].cpu().numpy(),
                'gcc_lmi': batch_output['gcc_lmi'][i, :seq_len, :seq_len].cpu().numpy(),
                'shp': batch_output['shp'][i, :seq_len].cpu().numpy()
            }
            
            if 'ca_dist' in batch_output:
                result['ca_dist'] = batch_output['ca_dist'][i, :seq_len, :seq_len].cpu().numpy()
                
            results.append(result)
            
        return results
    
    def process_files(self, filenames: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process multiple protein files and return their dynamics predictions.
        
        Args:
            filenames: List of PDB or sequence files to process
            
        Returns:
            Dictionary mapping protein IDs to their prediction results
        """
        results = {}
        
        # Process files in batches
        for batch_start in range(0, len(filenames), self.batch_size):
            batch_files = filenames[batch_start:batch_start + self.batch_size]
            
            # Load inputs
            inputs = []
            file_ids = []
            
            for filename in batch_files:
                try:
                    seq_feats, struct_feats = self._load_input(filename)
                    inputs.append({
                        'seq_feats': seq_feats,
                        'struct_feats': struct_feats
                    })
                    
                    # Use filename as ID (without extension)
                    file_id = os.path.splitext(os.path.basename(filename))[0]
                    file_ids.append(file_id)
                    
                except Exception as e:
                    logger.info(f"Error processing {filename}: {e}")
                    continue
            
            if not inputs:
                continue
                
            # Create and process batch
            batch = self._create_batch(inputs)
            batch_results = self._process_batch(batch)
            
            # Store results by file ID
            for file_id, result in zip(file_ids, batch_results):
                results[file_id] = result
                
        return results


class ProteomeProcessor:
    """
    Process entire proteomes or large protein sets with RocketSHP.
    """
    
    def __init__(
        self,
        model_checkpoint: str = "latest",
        output_dir: str = "proteome_predictions",
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = None
    ):
        """Initialize the proteome processor."""
        # Load model
        self.model = RocketSHP.load_from_checkpoint(model_checkpoint)
        
        # Create batch processor
        self.processor = BatchProcessor(
            model=self.model,
            batch_size=batch_size,
            device=device,
            use_structure=True  # Use structure when available for proteomes
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        
    def process_proteome(self, species: str = "human"):
        """
        Download and process a reference proteome.
        """
                # Map species to UniProt proteome ID
        proteome_map = {
            "human": ("UP000005640", "9606"),
            "mouse": ("UP000000589", "10090"),
            # Add more as needed
        }
        
        proteome_id, species_id = proteome_map.get(species, species)
        
        # Download proteome (in practice, you'd implement proper download logic)
        logger.info(f"Downloading {species} proteome ({proteome_id})...")
        url = f"https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/{proteome_id}/{proteome_id}_{species_id}.fasta.gz"
        
        import urllib.request
        import tempfile
        import gzip
        
        with tempfile.NamedTemporaryFile(suffix='.fasta.gz', delete=False) as temp_gz:
            urllib.request.urlretrieve(url, temp_gz.name)
            
            # Decompress to temporary file
            with gzip.open(temp_gz.name, 'rb') as f_in:
                with tempfile.NamedTemporaryFile(suffix='.fasta', delete=False) as temp_fasta:
                    import shutil
                    shutil.copyfileobj(f_in, temp_fasta)
                    fasta_path = temp_fasta.name
        
        # Process the proteome file (directly handling multiple sequences)
        results = self._process_multi_sequence_fasta(fasta_path)
        
        # Save results
        self._save_results(results, species)
        
        # Cleanup
        os.unlink(temp_gz.name)
        os.unlink(fasta_path)
        
        return results
    
    def _process_multi_sequence_fasta(self, fasta_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process a multi-sequence FASTA file.
        
        Args:
            fasta_path: Path to a FASTA file containing multiple sequences
            
        Returns:
            Dictionary mapping protein IDs to their prediction results
        """
        # Read all sequences from the FASTA file
        sequences = {}
        fasta_file = fasta.FastaFile.read(fasta_path)
        for header, sequence in fasta_file.items():
            # Extract protein ID from the header (usually the first part before the space)
            protein_id = header.split()[0]
            sequences[protein_id] = sequence
        
        print(f"Found {len(sequences)} sequences in {fasta_path}")
        
        # Process sequences in batches
        results = {}
        batch_proteins = []
        batch_sequences = []
        batch_ids = []
        
        for protein_id, sequence in sequences.items():
            batch_proteins.append(protein_id)
            batch_sequences.append(sequence)
            batch_ids.append(protein_id)
            
            # Process when batch is full
            if len(batch_sequences) >= self.processor.batch_size:
                batch_results = self._process_sequence_batch(batch_sequences, batch_ids)
                results.update(batch_results)
                
                # Reset batch
                batch_proteins = []
                batch_sequences = []
                batch_ids = []
        
        # Process any remaining sequences
        if batch_sequences:
            batch_results = self._process_sequence_batch(batch_sequences, batch_ids)
            results.update(batch_results)
            
        return results
    
    def _process_sequence_batch(self, sequences: List[str], 
                               protein_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process a batch of sequences directly.
        
        Args:
            sequences: List of amino acid sequences
            protein_ids: List of protein identifiers
            
        Returns:
            Dictionary mapping protein IDs to their prediction results
        """
        # Create inputs for batch processor
        inputs = []
        
        for sequence in sequences:
            seq_feats = load_sequence(sequence)
            inputs.append({
                'seq_feats': seq_feats,
                'struct_feats': None  # No structure for proteome sequences
            })
        
        # Create and process batch
        batch = self.processor._create_batch(inputs)
        batch_results = self.processor._process_batch(batch)
        
        # Map results to protein IDs
        results = {}
        for protein_id, result in zip(protein_ids, batch_results):
            results[protein_id] = result
            
        return results
    
    def process_from_files(self, file_list: Union[str, List[str]]):
        """
        Process proteins from a list of files or a file containing filenames.
        
        Args:
            file_list: Either a list of filenames or a path to a text file with one filename per line
        """
        # Handle case where file_list is a string (path to a file containing filenames)
        if isinstance(file_list, str):
            if os.path.isfile(file_list):
                with open(file_list, 'r') as f:
                    filenames = [line.strip() for line in f if line.strip()]
            else:
                # Check if this is a multi-sequence FASTA
                if file_list.endswith(('.fasta', '.fa')):
                    try:
                        fasta_file = fasta.FastaFile.read(file_list)
                        if len(fasta_file) > 1:
                            # This is a multi-sequence FASTA
                            return self._process_multi_sequence_fasta(file_list)
                    except Exception:
                        pass
                # Treat as a single file
                filenames = [file_list]
        else:
            filenames = file_list
            
        # Process using BatchProcessor
        print(f"Processing {len(filenames)} proteins...")
        results = self.processor.process_files(filenames)
        
        # Save results
        self._save_results(results, "custom")
        
        return results
    
    def _save_results(self, results, dataset_name):
        """Save results to output directory."""
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save each protein's results separately
        for protein_id, result in results.items():
            protein_file = dataset_dir / f"{protein_id}.npz"
            np.savez_compressed(
                protein_file,
                rmsf=result['rmsf'],
                gcc_lmi=result['gcc_lmi'],
                shp=result['shp']
            )
            
        # Save a metadata file
        with open(dataset_dir / "metadata.txt", "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total proteins: {len(results)}\n")
            f.write(f"Date processed: {datetime.datetime.now().isoformat()}\n")
            
        print(f"Results saved to {dataset_dir}")