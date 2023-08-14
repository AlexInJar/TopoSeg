import torch
import torch.nn as nn
from torch.nn import MSELoss
import ot
from ot.lp import wasserstein_1d
import numpy as np

from ripser import ripser
from scipy import sparse

from .topology import UnionFind
from swinunet_transform.swinublock import SwinTransformerSys

device = "cuda" if torch.cuda.is_available() else "cpu"

def lower_star_img(img):
    """
    Construct a lower star filtration on an image

    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data

    Returns
    -------
    I: ndarray (K, 2)
        A 0-dimensional persistence diagram corresponding to the sublevelset filtration
    """
    img = img.cpu().detach().numpy()
    m, n = img.shape

    idxs = np.arange(m * n).reshape((m, n))

    I = idxs.flatten()
    J = idxs.flatten()
    V = img.flatten()

    # Connect 8 spatial neighbors
    tidxs = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tidxs[1:-1, 1:-1] = idxs

    tD = np.ones_like(tidxs) * np.nan
    tD[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            thisJ = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
            thisD = np.roll(np.roll(tD, di, axis=0), dj, axis=1)
            thisD = np.maximum(thisD, tD)

            # Deal with boundaries
            boundary = ~np.isnan(thisD)
            thisI = tidxs[boundary]
            thisJ = thisJ[boundary]
            thisD = thisD[boundary]

            I = np.concatenate((I, thisI.flatten()))
            J = np.concatenate((J, thisJ.flatten()))
            V = np.concatenate((V, thisD.flatten()))
    sparseDM = sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))

    return ripser(sparseDM, distance_matrix=True, maxdim=1)["dgms"]


def get_persistence_dim1(img):
    '''
    return 1d array
    '''
    # print(" sum img: ",torch.sum(img) )
    if torch.sum(img) == 0:
        return np.array([1.])
    else:
        dgm_1 = lower_star_img(img)[1]
        bn_ar = dgm_1[dgm_1[:, 1] > 0.9][:, 0]
        bn_ar = bn_ar[bn_ar > 0.0]
        # else:
        # print("bn_ar: ", bn_ar)
        return bn_ar


def loss_2_img(img_i, img_j):
    '''
    img_i: pred
    img_j: gt
    '''
    # print("img_i.shape  ",img_i.shape)
    # print(img_i.shape)
    pd_i_np, pd_j_np = get_persistence_dim1(img_i), get_persistence_dim1(img_j)
    # print(torch.tensor(pd_i_np))
    # print("Pd_i_np_1d:", pd_i_np_1d)
    if len(pd_i_np) == 0:
        pd_i_1d = torch.tensor([1.0]).to(device=device)
    else:
        pd_i_1d = torch.cat([img_i[(img_i == i)] for i in pd_i_np]).to(
            device=device)  ## pred has back_prop, retrieve all backward function from img_i

    if len(pd_j_np) == 0:
        pd_j_1d = torch.tensor([1.0]).to(device=device)
    else:
        pd_j_1d = torch.tensor(pd_j_np).to(device=device)  ## gt has no back_prop, use numpy

    # print("pd_i_1d", pd_i_1d)
    # print("Pd_i_np.shape : {}".format(pd_i_1d.shape[0]))
    a, b = (torch.ones(pd_i_1d.shape[0]) / pd_i_1d.shape[0]).to(device=device), (
                torch.ones(pd_j_1d.shape[0]) / pd_j_1d.shape[0]).to(device=device)
    # print("a" , a)
    try:
        loss = wasserstein_1d(
            pd_i_1d,
            pd_j_1d,
            a,
            b
        )
        return loss
    except Exception as e:
        print("pd_i_1d", pd_i_1d)
        print("pd_j_1d", pd_j_1d)

class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles
        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        # print(((signature1 - signature2)**2))
        # print("sigerror",((signature1 - signature2)**2).sum(dim=-1))
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # print("sig1",sig1)
            # print("sig2",sig2)
            distance = self.sig_error(sig1, sig2)
            # print("distance", distance)
        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            # distance_components['metrics.distance1-2'] = distance1_2
            # distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            # distance_components['metrics.distance1-2'] = distance1_2
            # distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        # return distance, distance_components
        return distance

class PersistentHomologyCalculation:
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs), np.array([])


    # def encode(self, x):
    #     return self.autoencoder.encode(x)

    # def decode(self, z):
    #     return self.autoencoder.decode(z)  

class loss(nn.Module):
    def __init__(self,comp_top,comp_tae,comp_cnct,tau, lmda=5e-2,lmda2=2*10**(-4),lmda3=1):
        super(loss,self).__init__()
        self.reconst_loss = MSELoss()
        self.lmda = lmda
        self.lmda2 = lmda2
        self.lmda3 = lmda3
        self.comp_top = comp_top
        self.comp_tae = comp_tae
        self.comp_cnct = comp_cnct
        self.tau = tau
        self.swin_unet = SwinTransformerSys(img_size=256,in_chans=2,num_classes=2)
        self.topo_sig = TopologicalSignatureDistance()
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),requires_grad=True)
    #{
        #  def __init__(self, img_size=256, num_classes=2, zero_head=False, vis=False):
        # """SwinUnetAutoencoder constructor."""
        # super(SwinUnetAutoencoder, self).__init__()
        # self.num_classes = num_classes
        # # self.zero_head = zero_head
        
        # # self.pred_head = Sigmoid()
        
        # self.swin_unet = SwinTransformerSys(
        #     img_size=256,
        #     in_chans=2,
        #     num_classes=num_classes
        # )

        """Topologically Regularized Autoencoder."""
    # def __init__(self, lam=1., autoencoder_model='SwinUnetAutoencoder',
    #             ae_kwargs=None, toposig_kwargs=None):
    # lam: Regularization strength
    # ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
    # toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        # self.lam = lam
        # ae_kwargs = ae_kwargs if ae_kwargs else {}
        # toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        # self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        # self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)
        # self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
        #                                       requires_grad=True)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        #{
        # x_flat = x_flat.to(x.device)
        # }
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances
    
    
    def tae_loss(self, x, latent):

    # def encode(self, x):
    #     """Compute the latent representation using the SwinUnet encoder."""
    #     return self.swin_unet.forward_features(x)[0]

    # def decode(self, z):
    #     """Compute the reconstruction using the SwinUnet decoder."""
    #     return self.swin_unet.forward_up_features(z, None)
    
        x_distances = self._compute_distance_matrix(x)
        dimensions = x.size()

        # if len(dimensions) == 4:
        # If we have an image dataset, normalize using theoretical maximum
        batch_size, ch, b, w = dimensions
        # Compute the maximum distance we could get in the data space (this
        # is only valid for images wich are normalized between -1 and 1)
        max_distance = (2**2 * ch * b * w) ** 0.5
        x_distances = x_distances / max_distance
        # else:
        #     # Else just take the max distance we got in the batch
        #     x_distances = x_distances / x_distances.max()
        #{
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1, device=latent.device), requires_grad=True)
        # }
        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        # # Use reconstruction loss of autoencoder
        # ae_loss, ae_loss_comp = self.autoencoder(x)
        # print(latent_distances.size)
        # print(latent_distances)
        # print(x_distances)
        topo_error = self.topo_sig(x_distances, latent_distances)
        # print(topo_error)
        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
        return topo_error
    #}

    def topo_loss(self, inp, targ):
        # print("Input Shape: ",inp.shape)
        losses = []
        for image_pair in zip(inp, targ):
            image_1, image_2 = image_pair
            # Compute loss for this image pair
            image_1, image_2 = image_1.squeeze(), image_2.squeeze()
            loss = loss_2_img(image_1, image_2)
            # print("2_img_loss: ",loss)
            # Append loss to list of losses
            losses.append(loss)
        # Compute mean of losses for the batch
        batch_loss = torch.mean(torch.stack(losses))
        return batch_loss
    
    def cnct_loss(self, outs, targets):
        # Compute the condition tensor
        condition = (targets - self.tau) * (outs - self.tau)
        # Apply the indicator function element-wise
        indicator = torch.where(condition <= 0, torch.ones_like(condition), torch.zeros_like(condition))
        # Compute the sum of indicator values
        cnct_loss = indicator.sum()
        return  cnct_loss
    
    def forward(self, outs, samples, latent, targets):
        reconst_loss = self.reconst_loss(outs, targets)
        tae_loss = self.tae_loss(samples, latent) if self.comp_tae else 0
        topo_loss = self.topo_loss(outs, targets) if self.comp_top else 0
        cnct_loss = self.cnct_loss(outs, targets) if self.comp_cnct else 0
        # topo_loss = torch.tensor([0]).to(device=device)

        return (reconst_loss + self.lmda * topo_loss + self.lmda2 * tae_loss + self.lmda3 * cnct_loss,
                reconst_loss.item(),
                topo_loss,
                tae_loss,
                cnct_loss
                )


if __name__ == "__main__":
    # test_metric = topo_loss()

    nucinnd = np.load('../imgs/train/2452_3/nucinnd.npy')
    nucinnd_1 = np.load('../imgs/train/2452_3/nucinnd.npy')
    nucinnd.shape
    pred = torch.Tensor(nucinnd).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    pred.requires_grad_()
    # pred[:,:,50,78] = 0.8
    # pred[:,:,50,77] = 0.7
    # pred[:,:,50,76] = 0.6
    gtrth = torch.Tensor(nucinnd_1).unsqueeze(dim=0).unsqueeze(dim=0).cuda()

    ls = loss()
    los_val = ls.forward(pred, gtrth)

    print(
        "The loss between the {} and {} is {}".format("Pred", "Gtruch", los_val)
    )
    
   